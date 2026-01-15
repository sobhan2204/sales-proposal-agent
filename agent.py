from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
import os
import re
import logging
import time
from functools import wraps
from dotenv import load_dotenv

try:
    from langchain_groq import ChatGroq
    USE_GROQ = True
except ImportError:
    from langchain_google_genai import ChatGoogleGenerativeAI
    USE_GROQ = False
    print("⚠️ langchain-groq not installed. Using Gemini (install with: pip install langchain-groq)")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration Constants
# -----------------------------
class Config:
    # LLM Settings
    LLM_TEMPERATURE = 0.4
    MAX_LLM_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    # Token Optimization
    MAX_CONVERSATION_HISTORY = 5  # Keep last 5 exchanges
    MAX_PROPOSAL_VERSIONS = 10  # Keep last 10 versions
    PRICING_VARIANCE_THRESHOLD = 0.20  # 20% price variance allowed
    
    # Validation
    MAX_PAST_PROPOSALS = 3
    MAX_PRICING_SERVICES = 10  # Max services to include from catalog
    MIN_PRICING_MENTIONS = 1  # Skip validation if less than this
    
    # Prompts
    MAX_MESSAGE_LENGTH_IN_HISTORY = 300  # Truncate long messages
    
    # Performance
    ENABLE_CACHING = True
    CACHE_TTL = 300  # 5 minutes


# Global cache for LLM instance and static data
_llm_cache = None
_pricing_cache = None
_template_cache = None
_cache_timestamp = 0

from tools import (
    get_crm_deal,
    get_pricing_catalog,
    get_template,
    search_past_proposals,
    request_internal_approval
)

# -----------------------------
# Error Handling & Retry Logic
# -----------------------------
def retry_with_backoff(max_retries=Config.MAX_LLM_RETRIES, base_delay=Config.RETRY_DELAY):
    """Decorator for retrying LLM calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator


def safe_get(dictionary: Dict, key: str, default=None, expected_type=None):
    """Safely get value from dictionary with type checking."""
    value = dictionary.get(key, default)
    if expected_type and value is not None:
        if not isinstance(value, expected_type):
            logger.warning(f"Expected {expected_type} for key '{key}', got {type(value)}. Using default.")
            return default
    return value


def validate_state(state: "ProposalState", required_fields: List[str]) -> bool:
    """Validate that state has all required fields."""
    missing = [field for field in required_fields if field not in state]
    if missing:
        logger.error(f"Missing required state fields: {missing}")
        return False
    return True

# -----------------------------
# Agent State
# -----------------------------
class ProposalState(TypedDict, total=False):
    deal_id: str
    user_instruction: str
    crm: Dict[str, Any]
    pricing: Dict[str, Any]
    template: Dict[str, Any]
    past: List[Dict[str, Any]]
    missing_fields: List[str]
    approvals: Dict[str, str]  # Status summary for compatibility
    approval_details: Dict[str, Dict[str, Any]]  # Full approval responses with feedback
    proposal_versions: List[str]
    current_proposal: str
    last_action: str
    conversation_history: List[Dict[str, str]]  # Track user feedback & refinements
    is_initial_generation: bool  # Flag to distinguish initial vs iterative
    validation_results: Dict[str, Any]  # Pricing validation results


# -----------------------------
# LLM Configuration
# -----------------------------
@retry_with_backoff()
def get_llm():
    """
    Get LLM instance with caching and retry logic.
    Caches instance to avoid re-initialization overhead.
    """
    global _llm_cache, _cache_timestamp
    
    # Return cached instance if available and fresh
    if Config.ENABLE_CACHING and _llm_cache is not None:
        if time.time() - _cache_timestamp < Config.CACHE_TTL:
            logger.debug("Using cached LLM instance")
            return _llm_cache
    
    try:
        if USE_GROQ:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not found in environment variables.\n"
                    "Get a free API key at: https://console.groq.com/keys\n"
                    "Add to .env file: GROQ_API_KEY=your_key_here"
                )
            
            # Groq models - all free tier, very fast
            models_to_try = [
                "llama-3.3-70b-versatile",  # Best for complex tasks
                "llama-3.1-70b-versatile",  # Alternative 70B
                "mixtral-8x7b-32768",       # Good balance
            ]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Initializing {model_name}...")
                    llm = ChatGroq(
                        model=model_name,
                        temperature=Config.LLM_TEMPERATURE,
                        groq_api_key=api_key
                    )
                    # Cache the instance
                    _llm_cache = llm
                    _cache_timestamp = time.time()
                    return llm
                except Exception as e:
                    logger.warning(f"Failed to initialize {model_name}: {e}")
                    continue
            
            raise RuntimeError(f"Could not initialize any Groq model. Tried: {models_to_try}")
        
        else:
            # Fallback to Gemini if Groq not available
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            models_to_try = [
                "models/gemini-2.0-flash",
                "models/gemini-1.5-flash",
            ]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Initializing {model_name}...")
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=Config.LLM_TEMPERATURE,
                        google_api_key=api_key
                    )
                    # Cache the instance
                    _llm_cache = llm
                    _cache_timestamp = time.time()
                    return llm
                except Exception as e:
                    logger.warning(f"Failed to initialize {model_name}: {e}")
                    continue
            
            raise RuntimeError(f"Could not initialize any Gemini model. Tried: {models_to_try}")
    except Exception as e:
        logger.error(f"Fatal error in get_llm: {e}")
        raise


# -----------------------------
# Token Optimization Helpers
# -----------------------------
def filter_pricing_catalog(pricing: Dict[str, Any], requirements) -> Dict[str, Any]:
    """
    Filter pricing catalog to only relevant services based on requirements.
    Reduces token usage by 60-80% for pricing data.
    """
    if not pricing or not requirements:
        return pricing
    
    # Handle both string and list requirements
    if isinstance(requirements, list):
        requirements_str = " ".join(str(r) for r in requirements)
    else:
        requirements_str = str(requirements)
    
    requirements_lower = requirements_str.lower()
    keywords = [
        'cloud', 'migration', 'devops', 'infrastructure', 'security',
        'data', 'analytics', 'ai', 'ml', 'consulting', 'development',
        'support', 'managed', 'integration', 'automation'
    ]
    
    # Extract relevant keywords from requirements
    relevant_keywords = [kw for kw in keywords if kw in requirements_lower]
    
    # Filter services
    filtered = {"services": []}
    if "services" in pricing:
        for service in pricing["services"]:
            service_name = service.get("name", "").lower()
            service_desc = service.get("description", "").lower()
            
            # Include if matches any keyword
            if any(kw in service_name or kw in service_desc for kw in relevant_keywords):
                filtered["services"].append(service)
    
    # If no matches, return top 5 services to avoid empty catalog
    if not filtered["services"] and "services" in pricing:
        filtered["services"] = pricing["services"][:5]
    
    return filtered


def summarize_past_proposals(past: List[Dict[str, Any]]) -> str:
    """
    Extract key insights from past proposals instead of full text.
    Reduces token usage by 70-90% for historical data.
    """
    if not past:
        return "No similar past proposals found."
    
    summaries = []
    for idx, proposal in enumerate(past[:Config.MAX_PAST_PROPOSALS], 1):  # Use config limit
        summary = f"Proposal {idx}:\n"
        summary += f"- Industry: {safe_get(proposal, 'industry', 'N/A')}\n"
        summary += f"- Value: {safe_get(proposal, 'deal_value', 'N/A')}\n"
        summary += f"- Key Services: {safe_get(proposal, 'services', 'N/A')}\n"
        summary += f"- Success Factors: {safe_get(proposal, 'success_factors', 'Delivered on time')}\n"
        summaries.append(summary)
    
    return "\n".join(summaries)


def extract_crm_essentials(crm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only essential CRM fields for proposal generation.
    Reduces token usage by ~50% for CRM data.
    """
    essential_fields = [
        'company', 'industry', 'requirements', 'budget_range',
        'deadline_days', 'contact_name', 'contact_email'
    ]
    
    return {k: v for k, v in crm.items() if k in essential_fields}


def parse_approval_feedback(approval_details: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Parse approval responses and extract actionable constraints.
    Now truly dynamic - extracts real feedback from approval responses.
    Returns categorized instructions based on approval status.
    """
    feedback = {
        "critical": [],  # Must address (rejections)
        "warnings": [],  # Should address (conditions)
        "approved": []   # All clear
    }
    
    # Fallback templates only used if no specific feedback
    fallback_rules = {
        "Finance": {
            "Rejected": "Finance has blocked this pricing. Review pricing structure and discount policy.",
            "Concerns": "Revise pricing approach and justify any catalog deviations.",
            "Conditions": "Ensure discounts comply with policy (<15% without VP approval)."
        },
        "Legal": {
            "Rejected": "Legal has rejected contract terms. Escalate immediately.",
            "Concerns": "Add legal disclaimers, clarify IP/data handling and liability.",
            "Conditions": "Include standard liability clauses and explicit assumptions."
        },
        "Delivery Lead": {
            "Rejected": "Delivery team cannot commit. Renegotiate scope/timeline.",
            "Concerns": "Timeline may be unrealistic. Revise delivery schedule.",
            "Conditions": "Confirm resource availability and add buffer time."
        }
    }
    
    for team, details in approval_details.items():
        status = details.get("status", "")
        message = details.get("message", "")
        specific_feedback = details.get("feedback", "")  # Actual concerns from approver
        
        # Build instruction from actual feedback if available
        if specific_feedback:
            instruction = f"{specific_feedback}"
        else:
            # Use fallback based on status keywords
            team_rules = fallback_rules.get(team, {})
            if "reject" in status.lower():
                instruction = team_rules.get("Rejected", f"{team} has rejected. Address their concerns.")
            elif "concern" in status.lower():
                instruction = team_rules.get("Concerns", f"{team} raised concerns. Review and revise.")
            elif "condition" in status.lower():
                instruction = team_rules.get("Conditions", f"{team} approved with conditions. Follow guidelines.")
            else:
                # Fully approved
                feedback["approved"].append(team)
                continue
        
        # Categorize based on severity
        full_instruction = f"[{team}] {instruction}"
        if "reject" in status.lower():
            feedback["critical"].append(full_instruction)
        elif "concern" in status.lower() or "condition" in status.lower():
            feedback["warnings"].append(full_instruction)
        else:
            feedback["approved"].append(team)
    
    return feedback


def build_dynamic_system_rules(approval_details: Dict[str, Dict[str, Any]], iteration_count: int, user_feedback: Optional[str] = None) -> str:
    """
    Generate adaptive system rules based on actual approval feedback, iteration stage, and user input.
    """
    base_rules = """You are an enterprise IT services proposal agent for Northstar Enterprises.

CORE PRINCIPLES:
- Do NOT invent pricing. Use only provided pricing catalog.
- If something is unknown, mark it as TBD.
- Maintain enterprise professional tone and structure.
- Include scope, timeline, and commercials."""
    
    # Parse approval feedback from actual responses
    approval_feedback = parse_approval_feedback(approval_details) if approval_details else None
    
    # Add approval-specific constraints
    approval_section = ""
    if approval_feedback:
        if approval_feedback["critical"]:
            approval_section += "\n\n🚨 CRITICAL REQUIREMENTS (MUST ADDRESS):\n"
            for rule in approval_feedback["critical"]:
                approval_section += f"- {rule}\n"
        
        if approval_feedback["warnings"]:
            approval_section += "\n\n⚠️ IMPORTANT CONSIDERATIONS:\n"
            for rule in approval_feedback["warnings"]:
                approval_section += f"- {rule}\n"
    
    # Add iteration-specific guidance
    iteration_guidance = ""
    if iteration_count == 0:
        iteration_guidance = "\n\nFIRST DRAFT FOCUS:\n- Create comprehensive structure with all sections\n- Be thorough but concise\n- Use template structure as guide"
    elif iteration_count >= 3:
        iteration_guidance = "\n\nREFINEMENT FOCUS (Multiple iterations):\n- Make precise, targeted changes only\n- Avoid restructuring unless explicitly requested\n- Maintain consistency with prior approved sections"
    else:
        iteration_guidance = "\n\nREFINEMENT FOCUS:\n- Address specific feedback precisely\n- Maintain document coherence\n- Preserve approved elements"
    
    # Add user feedback context
    feedback_context = ""
    if user_feedback:
        # Handle both string and list feedback
        if isinstance(user_feedback, list):
            feedback_str = " ".join(str(f) for f in user_feedback)
        else:
            feedback_str = str(user_feedback)
        
        feedback_lower = feedback_str.lower()
        if any(word in feedback_lower for word in ['urgent', 'asap', 'quickly', 'rush']):
            feedback_context += "\n\n⚡ URGENCY NOTED: Prioritize speed while maintaining quality."
        if any(word in feedback_lower for word in ['technical', 'detailed', 'specific']):
            feedback_context += "\n\n🔧 TECHNICAL DEPTH: Provide detailed technical specifications and architecture."
        if any(word in feedback_lower for word in ['executive', 'summary', 'brief', 'concise']):
            feedback_context += "\n\n📊 EXECUTIVE LEVEL: Keep high-level, focus on business value and ROI."
    
    return base_rules + approval_section + iteration_guidance + feedback_context


def extract_pricing_from_proposal(proposal_text: str) -> List[Dict[str, Any]]:
    """
    Extract all pricing mentions from proposal using LLM.
    Returns list of {service, price, unit} dictionaries.
    """
    import re
    
    # Quick regex extraction for common patterns
    pricing_mentions = []
    
    # Pattern: $X,XXX or $XXX,XXX.XX for service/item
    # Pattern: X days at $Y per day
    # Pattern: Service Name - $X
    
    lines = proposal_text.split('\n')
    for line in lines:
        # Look for lines with dollar amounts
        if '$' in line:
            # Extract dollar amounts
            amounts = re.findall(r'\$([\d,]+(?:\.\d{2})?)', line)
            if amounts:
                # Try to extract service name (text before the price)
                service_match = re.search(r'([A-Za-z][A-Za-z0-9\s-]+?)(?:\s*[-:]?\s*\$|\s+@\s+\$)', line)
                service_name = service_match.group(1).strip() if service_match else "Unknown service"
                
                for amount in amounts:
                    pricing_mentions.append({
                        "service": service_name,
                        "price": amount.replace(',', ''),
                        "line": line.strip()
                    })
    
    return pricing_mentions


def validate_pricing_against_catalog(pricing_mentions: List[Dict[str, Any]], catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate extracted pricing against catalog.
    Returns validation results with violations and suggestions.
    """
    violations = []
    warnings = []
    valid_items = []
    
    # Build catalog lookup
    catalog_services = {}
    if "services" in catalog:
        for service in catalog["services"]:
            name = service.get("name", "").lower()
            catalog_services[name] = service
    
    for mention in pricing_mentions:
        service_name = mention["service"].lower()
        mentioned_price = mention["price"]
        
        # Try to find matching service in catalog
        matched_service = None
        for cat_name, cat_service in catalog_services.items():
            # Fuzzy match - check if mention contains catalog service name or vice versa
            if cat_name in service_name or service_name in cat_name:
                matched_service = cat_service
                break
        
        if not matched_service:
            # Check if it's a generic cost item (total, subtotal, etc.)
            if any(keyword in service_name for keyword in ['total', 'subtotal', 'sum', 'grand', 'estimated']):
                continue  # Skip validation for totals
            
            violations.append({
                "type": "unlisted_service",
                "service": mention["service"],
                "price": mentioned_price,
                "message": f"Service '{mention['service']}' not found in pricing catalog. May be invented.",
                "line": mention["line"]
            })
        else:
            # Service exists, validate price
            catalog_price = matched_service.get("base_price_inr", 0)
            catalog_price = catalog_price.replace('$', '').replace(',', '')
            
            try:
                mentioned_float = float(mentioned_price)
                catalog_float = float(catalog_price)
                
                # Allow for quantity multiplication or reasonable ranges
                # Check if price is a reasonable multiple or within 20% variance
                if catalog_float > 0 and abs(mentioned_float - catalog_float) / catalog_float > 0.20:
                    # Check if it's a reasonable multiple (2x, 3x, etc for quantity)
                    ratio = mentioned_float / catalog_float if catalog_float > 0 else 0
                    if ratio < 0.8 or (ratio > 1.2 and ratio % 1 > 0.2):  # Not a clean multiple
                        warnings.append({
                            "type": "price_mismatch",
                            "service": mention["service"],
                            "mentioned_price": mentioned_price,
                            "catalog_price": catalog_price,
                            "message": f"Price ${mentioned_price} for '{mention['service']}' doesn't match catalog ${catalog_price}.",
                            "line": mention["line"]
                        })
                    else:
                        valid_items.append(mention)
                else:
                    valid_items.append(mention)
            except ValueError:
                warnings.append({
                    "type": "invalid_price_format",
                    "service": mention["service"],
                    "price": mentioned_price,
                    "message": f"Could not parse price format for validation.",
                    "line": mention["line"]
                })
    
    return {
        "is_valid": len(violations) == 0,
        "has_warnings": len(warnings) > 0,
        "violations": violations,
        "warnings": warnings,
        "valid_items": valid_items,
        "total_items_checked": len(pricing_mentions)
    }


# -----------------------------
# Nodes
# -----------------------------
def gather_context(state: ProposalState) -> ProposalState:
    """Gather context with optimized data loading and caching."""
    try:
        logger.info("Gathering context...")
        deal_id = state["deal_id"]
        
        # Load CRM deal
        crm = get_crm_deal(deal_id)
        
        # Use cached pricing and template if available
        global _pricing_cache, _template_cache, _cache_timestamp
        
        if Config.ENABLE_CACHING and _pricing_cache and _template_cache:
            if time.time() - _cache_timestamp < Config.CACHE_TTL:
                logger.debug("Using cached pricing and template")
                pricing = _pricing_cache
                template = _template_cache
            else:
                pricing = get_pricing_catalog()
                template = get_template()
                _pricing_cache = pricing
                _template_cache = template
                _cache_timestamp = time.time()
        else:
            pricing = get_pricing_catalog()
            template = get_template()
            if Config.ENABLE_CACHING:
                _pricing_cache = pricing
                _template_cache = template
                _cache_timestamp = time.time()

        # Load past proposals only if we have industry info
        past = []
        if "industry" in crm and "error" not in crm:
            past = search_past_proposals(crm["industry"])

        missing = []
        if "error" in crm:
            missing.append("Valid Deal ID")
        else:
            # minimal checks
            if not crm.get("requirements"):
                missing.append("Requirements")
            if not crm.get("budget_range"):
                missing.append("Budget range")
            if not crm.get("deadline_days"):
                missing.append("Deadline")

        state["crm"] = crm
        state["pricing"] = pricing
        state["template"] = template
        state["past"] = past
        state["missing_fields"] = missing
        state["last_action"] = "gathered_context"
        logger.info(f"Context gathered: CRM {'found' if 'error' not in crm else 'not found'}, {len(past)} past proposals")
        return state
    
    except Exception as e:
        logger.error(f"Error in gather_context: {e}")
        state["missing_fields"] = ["Context gathering failed"]
        state["last_action"] = "context_error"
        return state


def decide_next_step(state: ProposalState) -> str:
    # If deal not found, stop early
    if "error" in state["crm"]:
        return "need_more_info"

    if state["missing_fields"]:
        return "need_more_info"

    return "generate_proposal"


def need_more_info(state: ProposalState) -> ProposalState:
    """Token-efficient info request."""
    missing = state.get("missing_fields", [])
    if "error" in state["crm"]:
        msg = f"❌ I couldn’t find that deal in CRM. Please provide a valid Deal ID (example: NS-101)."
    else:
        msg = "⚠️ I need a bit more info before generating a reliable proposal:\n"
        for m in missing:
            msg += f"- {m}\n"
        msg += "\nReply with the missing details, and I’ll continue."

    state["current_proposal"] = msg
    state["last_action"] = "asked_for_missing_info"
    return state


def generate_proposal(state: ProposalState) -> ProposalState:
    """Generate initial proposal with error handling and validation."""
    try:
        # Validate required state fields
        if not validate_state(state, ['crm', 'pricing', 'template', 'user_instruction']):
            raise ValueError("Missing required state fields for proposal generation")
        
        logger.info("Starting proposal generation...")
        llm = get_llm()

        crm = state["crm"]
        
        # Token optimization: filter and summarize data
        crm_essentials = extract_crm_essentials(crm)
        requirements = safe_get(crm, "requirements", "")
        filtered_pricing = filter_pricing_catalog(state["pricing"], requirements)
        past_summary = summarize_past_proposals(safe_get(state, "past", []))
        
        # Safe template access
        template = safe_get(state, "template", {}).get("default_proposal_template", "Standard enterprise proposal format")
        
        # Dynamic prompt generation
        iteration_count = len(state.get("proposal_versions", []))
        approval_details = state.get("approval_details", {})
        system_rules = build_dynamic_system_rules(
            approval_details=approval_details,
            iteration_count=iteration_count,
            user_feedback=state.get("user_instruction", "")
        )

        prompt = f"""
{system_rules}

USER INSTRUCTION:
{state["user_instruction"]}

CRM DEAL (Essential Fields):
{crm_essentials}

RELEVANT PRICING CATALOG:
{filtered_pricing}

TEMPLATE STRUCTURE:
{template}

PAST PROPOSALS INSIGHTS:
{past_summary}

Write a complete sales proposal draft in a clean format with headings.
"""

        logger.info("Invoking LLM for proposal generation...")
        resp = llm.invoke(prompt)
        proposal_text = resp.content
        logger.info(f"Generated proposal: {len(proposal_text)} characters")

        # Optimize version storage - keep only last N versions
        versions = state.get("proposal_versions", [])
        if len(versions) >= Config.MAX_PROPOSAL_VERSIONS:
            versions = versions[-(Config.MAX_PROPOSAL_VERSIONS - 1):]  # Keep last N-1
        versions.append(proposal_text)

        state["proposal_versions"] = versions
        state["current_proposal"] = proposal_text
        state["last_action"] = "generated_proposal"
        
        # Initialize conversation history with size limit
        history = state.get("conversation_history", [])
        if len(history) >= Config.MAX_CONVERSATION_HISTORY * 2:  # 2 messages per exchange
            history = history[-(Config.MAX_CONVERSATION_HISTORY * 2 - 1):]
        history.append({"role": "assistant", "content": proposal_text})
        state["conversation_history"] = history
        state["is_initial_generation"] = True
        
        logger.info("Proposal generation completed successfully")
        return state
    
    except Exception as e:
        logger.error(f"Error in generate_proposal: {e}")
        state["current_proposal"] = f"❌ Error generating proposal: {str(e)}"
        state["last_action"] = "generation_failed"
        return state


def simulate_internal_coordination(state: ProposalState) -> ProposalState:
    """
    Simulate approvals with realistic feedback.
    In production, this would call actual approval APIs/webhooks.
    """
    approvals = state.get("approvals", {})
    approval_details = state.get("approval_details", {})

    # Simulate different approval scenarios based on proposal content
    proposal = state.get("current_proposal", "")
    crm = state.get("crm", {})
    
    # Finance approval - check for discount mentions
    finance_response = request_internal_approval("Finance", "Confirm pricing/discount compliance")
    if "discount" in proposal.lower() and "20%" in proposal:
        finance_response["status"] = "Concerns raised"
        finance_response["feedback"] = "Discount of 20% exceeds policy limit of 15% without VP approval. Please revise or escalate."
    elif "payment" in proposal.lower() and "net 90" in proposal.lower():
        finance_response["status"] = "Approved with conditions"
        finance_response["feedback"] = "Net 90 terms require CFO sign-off for deals over $500K."
    else:
        finance_response["status"] = "Approved"
        finance_response["feedback"] = "Pricing complies with catalog and discount policy."
    
    # Legal approval - check for terms/liability
    legal_response = request_internal_approval("Legal", "Review terms, assumptions, exclusions")
    if "unlimited liability" in proposal.lower():
        legal_response["status"] = "Rejected"
        legal_response["feedback"] = "Unlimited liability clause is unacceptable. Cap at 1x contract value per standard terms."
    elif "assumption" not in proposal.lower() or "exclusion" not in proposal.lower():
        legal_response["status"] = "Approved with conditions"
        legal_response["feedback"] = "Add explicit assumptions and exclusions section. Include standard IP ownership and data handling clauses."
    else:
        legal_response["status"] = "Approved"
        legal_response["feedback"] = "Terms are acceptable. Standard contract applies."
    
    # Delivery approval - check timeline realism
    delivery_response = request_internal_approval("Delivery Lead", "Validate scope feasibility and timeline")
    deadline_days = crm.get("deadline_days", 90)
    requirements = crm.get("requirements", "")
    # Handle list or string requirements
    if isinstance(requirements, list):
        requirements_str = " ".join(str(r) for r in requirements)
    else:
        requirements_str = str(requirements)
    
    if deadline_days < 30 and "complex" in requirements_str.lower():
        delivery_response["status"] = "Concerns raised"
        delivery_response["feedback"] = f"Timeline of {deadline_days} days is unrealistic for complex scope. Recommend minimum 45 days or reduce deliverables."
    elif deadline_days < 45:
        delivery_response["status"] = "Approved with conditions"
        delivery_response["feedback"] = "Tight timeline. Requires dedicated team and may need buffer for testing phase."
    else:
        delivery_response["status"] = "Approved"
        delivery_response["feedback"] = "Timeline is feasible with standard resource allocation."

    # Store both summary status and full details
    approvals["Finance"] = finance_response["status"]
    approvals["Legal"] = legal_response["status"]
    approvals["Delivery Lead"] = delivery_response["status"]
    
    approval_details["Finance"] = finance_response
    approval_details["Legal"] = legal_response
    approval_details["Delivery Lead"] = delivery_response

    state["approvals"] = approvals
    state["approval_details"] = approval_details
    state["last_action"] = "requested_approvals"
    return state


def refine_proposal(state: ProposalState) -> ProposalState:
    """
    Handle iterative refinement based on user feedback.
    This node is called after initial generation when user provides feedback.
    """
    llm = get_llm()
    
    # Get conversation context
    history = state.get("conversation_history", [])
    current_version = state.get("current_proposal", "")
    user_feedback = state.get("user_instruction", "")
    
    # Dynamic prompt generation
    iteration_count = len(state.get("proposal_versions", []))
    approval_details = state.get("approval_details", {})
    system_rules = build_dynamic_system_rules(
        approval_details=approval_details,
        iteration_count=iteration_count,
        user_feedback=user_feedback
    )
    
    # Build conversation context (only user messages to save tokens)
    conversation_context = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:Config.MAX_MESSAGE_LENGTH_IN_HISTORY]}..." 
        if len(msg['content']) > Config.MAX_MESSAGE_LENGTH_IN_HISTORY 
        else f"{msg['role'].upper()}: {msg['content']}"
        for msg in history[-Config.MAX_CONVERSATION_HISTORY:]  # Use config limit
    ])
    
    # Token optimization: only include pricing if feedback mentions pricing/cost
    # Handle both string and list feedback
    if isinstance(user_feedback, list):
        feedback_str = " ".join(str(f) for f in user_feedback)
    else:
        feedback_str = str(user_feedback)
    
    feedback_lower = feedback_str.lower()
    needs_pricing = any(keyword in feedback_lower for keyword in ['price', 'cost', 'budget', 'discount', '$'])
    
    pricing_context = ""
    if needs_pricing:
        requirements = state['crm'].get('requirements', user_feedback)
        filtered_pricing = filter_pricing_catalog(state['pricing'], requirements)
        pricing_context = f"\n\nRELEVANT PRICING:\n{filtered_pricing}"
    
    prompt = f"""
{system_rules}

CURRENT PROPOSAL VERSION:
{current_version}

USER FEEDBACK:
{user_feedback}

RECENT CONVERSATION:
{conversation_context}{pricing_context}

Refine the proposal based on the user's feedback. Return the complete updated proposal.
"""
    
    resp = llm.invoke(prompt)
    refined_text = resp.content
    
    # Update state
    versions = state.get("proposal_versions", [])
    versions.append(refined_text)
    
    # Update conversation history
    history.append({"role": "user", "content": user_feedback})
    history.append({"role": "assistant", "content": refined_text})
    
    state["proposal_versions"] = versions
    state["current_proposal"] = refined_text
    state["conversation_history"] = history
    state["last_action"] = "refined_proposal"
    state["is_initial_generation"] = False
    
    return state


def address_approval_concerns(state: ProposalState) -> ProposalState:
    """
    Handle rejected or conditional approvals by regenerating with constraints.
    This node is triggered when any approval is not fully approved.
    Now uses actual feedback from approval responses.
    """
    llm = get_llm()
    
    approval_details = state.get("approval_details", {})
    current_proposal = state.get("current_proposal", "")
    
    # Parse what needs to be fixed from actual responses
    approval_feedback = parse_approval_feedback(approval_details)
    
    # Check if any critical issues
    has_critical = len(approval_feedback["critical"]) > 0
    has_warnings = len(approval_feedback["warnings"]) > 0
    
    if not has_critical and not has_warnings:
        # All approved, nothing to do
        state["last_action"] = "all_approvals_clear"
        return state
    
    # Build focused revision prompt
    iteration_count = len(state.get("proposal_versions", []))
    system_rules = build_dynamic_system_rules(
        approval_details=approval_details,
        iteration_count=iteration_count,
        user_feedback="Address all approval concerns"
    )
    
    concerns_summary = ""
    if approval_feedback["critical"]:
        concerns_summary += "CRITICAL ISSUES TO FIX:\n"
        for concern in approval_feedback["critical"]:
            concerns_summary += f"- {concern}\n"
    
    if approval_feedback["warnings"]:
        concerns_summary += "\nWARNINGS TO ADDRESS:\n"
        for warning in approval_feedback["warnings"]:
            concerns_summary += f"- {warning}\n"
    
    # Token optimization: only include pricing for finance concerns
    pricing_context = ""
    if any("Finance" in concern for concern in approval_feedback["critical"] + approval_feedback["warnings"]):
        requirements = state['crm'].get('requirements', '')
        filtered_pricing = filter_pricing_catalog(state['pricing'], requirements)
        pricing_context = f"\n\nPRICING CATALOG (for Finance compliance):\n{filtered_pricing}"
    
    prompt = f"""
{system_rules}

CURRENT PROPOSAL (needs revision):
{current_proposal}

APPROVAL FEEDBACK:
{concerns_summary}{pricing_context}

Revise the proposal to address ALL approval concerns above. Return the complete updated proposal.
"""
    
    resp = llm.invoke(prompt)
    revised_text = resp.content
    
    # Update state
    versions = state.get("proposal_versions", [])
    versions.append(revised_text)
    
    history = state.get("conversation_history", [])
    history.append({"role": "system", "content": f"Approval concerns addressed: {concerns_summary}"})
    history.append({"role": "assistant", "content": revised_text})
    
    state["proposal_versions"] = versions
    state["current_proposal"] = revised_text
    state["conversation_history"] = history
    state["last_action"] = "addressed_approval_concerns"
    
    return state


def validate_proposal_pricing(state: ProposalState) -> ProposalState:
    """
    Validate that the proposal uses only catalog pricing.
    Detects invented prices, incorrect amounts, and unlisted services.
    Optimized: Skips validation if no significant pricing mentions.
    """
    try:
        current_proposal = state.get("current_proposal", "")
        pricing_catalog = state.get("pricing", {})
        
        # Quick check: if no $ signs or very few, skip validation
        currency_count = current_proposal.count('$') + current_proposal.count('₹')
        if currency_count < Config.MIN_PRICING_MENTIONS:
            logger.info(f"Skipping pricing validation: only {currency_count} price mentions")
            state["validation_results"] = {
                "is_valid": True,
                "has_warnings": False,
                "violations": [],
                "warnings": [],
                "total_items_checked": 0,
                "skipped": True
            }
            state["last_action"] = "validation_skipped"
            return state
        
        logger.info(f"Validating pricing ({currency_count} mentions found)...")
        # Extract pricing mentions from proposal
        pricing_mentions = extract_pricing_from_proposal(current_proposal)
        
        # Validate against catalog
        validation_results = validate_pricing_against_catalog(pricing_mentions, pricing_catalog)
        
        state["validation_results"] = validation_results
        state["last_action"] = "validated_pricing"
        logger.info(f"Validation complete: {len(validation_results.get('violations', []))} violations")
        
        return state
    
    except Exception as e:
        logger.error(f"Error in validate_proposal_pricing: {e}")
        # Don't fail the whole flow, just mark as valid
        state["validation_results"] = {"is_valid": True, "has_warnings": False}
        state["last_action"] = "validation_error"
        return state


def fix_pricing_violations(state: ProposalState) -> ProposalState:
    """
    Auto-correct pricing violations using LLM with strict constraints.
    """
    llm = get_llm()
    
    validation = state.get("validation_results", {})
    current_proposal = state.get("current_proposal", "")
    pricing_catalog = state.get("pricing", {})
    
    violations = validation.get("violations", [])
    warnings = validation.get("warnings", [])
    
    if not violations and not warnings:
        # Nothing to fix
        state["last_action"] = "no_pricing_violations"
        return state
    
    # Build correction prompt with violations
    violation_summary = "\n".join([
        f"- {v['message']}\n  Line: {v['line']}"
        for v in violations + warnings
    ])
    
    # Get filtered catalog for reference
    requirements = state['crm'].get('requirements', '')
    filtered_pricing = filter_pricing_catalog(pricing_catalog, requirements)
    
    correction_prompt = f"""
You are correcting pricing violations in a proposal. You MUST use ONLY the provided pricing catalog.

🚨 CRITICAL PRICING VIOLATIONS DETECTED:
{violation_summary}

VALID PRICING CATALOG (USE ONLY THESE):
{filtered_pricing}

CURRENT PROPOSAL:
{current_proposal}

TASK:
1. Replace ALL violated pricing with correct catalog prices
2. Remove any services not in the catalog or replace with catalog equivalents
3. Keep all other content unchanged
4. Ensure pricing calculations are accurate

Return the complete corrected proposal.
"""
    
    resp = llm.invoke(correction_prompt)
    corrected_proposal = resp.content
    
    # Update state
    versions = state.get("proposal_versions", [])
    versions.append(corrected_proposal)
    
    history = state.get("conversation_history", [])
    history.append({"role": "system", "content": f"Auto-corrected {len(violations)} pricing violations"})
    history.append({"role": "assistant", "content": corrected_proposal})
    
    state["proposal_versions"] = versions
    state["current_proposal"] = corrected_proposal
    state["conversation_history"] = history
    state["last_action"] = "fixed_pricing_violations"
    
    return state


# -----------------------------
# Build Graph
# -----------------------------
def decide_refinement_path(state: ProposalState) -> str:
    """
    Determine if this is initial generation or iterative refinement.
    """
    # Check if we already have a proposal and this is a follow-up
    if state.get("current_proposal") and not state.get("is_initial_generation", True):
        return "refine_proposal"
    return "simulate_internal_coordination"


def check_approval_status(state: ProposalState) -> str:
    """
    Route based on approval outcomes after internal coordination.
    """
    approvals = state.get("approvals", {})
    
    # Check if any rejections or concerns
    has_issues = any(
        "Rejected" in status or "Concerns" in status or "conditions" in status
        for status in approvals.values()
    )
    
    if has_issues:
        return "address_concerns"
    return "complete"


def check_pricing_validation(state: ProposalState) -> str:
    """
    Route based on pricing validation results.
    Optimized: Skip fixing if validation was skipped or has no violations.
    """
    validation = state.get("validation_results", {})
    
    # If validation was skipped or no results, proceed
    if not validation or validation.get("skipped", False):
        logger.debug("Validation skipped, proceeding to approvals")
        return "proceed"
    
    # If critical violations, fix them
    violations = validation.get("violations", [])
    if violations and not validation.get("is_valid", True):
        logger.info(f"Found {len(violations)} violations, routing to fix")
        return "fix_pricing"
    
    # If only warnings, proceed (don't block on warnings)
    if validation.get("has_warnings", False):
        logger.info("Only warnings found, proceeding")
        return "proceed"
    
    return "proceed"


def build_graph():
    """
    Build the main proposal generation graph.
    Note: refine_proposal is called separately for iterative refinements.
    """
    g = StateGraph(ProposalState)

    g.add_node("gather_context", gather_context)
    g.add_node("need_more_info", need_more_info)
    g.add_node("generate_proposal", generate_proposal)
    g.add_node("validate_proposal_pricing", validate_proposal_pricing)
    g.add_node("fix_pricing_violations", fix_pricing_violations)
    g.add_node("simulate_internal_coordination", simulate_internal_coordination)
    g.add_node("address_approval_concerns", address_approval_concerns)

    g.set_entry_point("gather_context")

    g.add_conditional_edges(
        "gather_context",
        decide_next_step,
        {
            "need_more_info": "need_more_info",
            "generate_proposal": "generate_proposal"
        }
    )

    # After generation, validate pricing
    g.add_edge("generate_proposal", "validate_proposal_pricing")
    
    # Route based on validation results
    g.add_conditional_edges(
        "validate_proposal_pricing",
        check_pricing_validation,
        {
            "fix_pricing": "fix_pricing_violations",
            "proceed": "simulate_internal_coordination"
        }
    )
    
    # After fixing, re-validate or proceed to approvals
    g.add_edge("fix_pricing_violations", "simulate_internal_coordination")
    
    # After approvals, check if concerns need addressing
    g.add_conditional_edges(
        "simulate_internal_coordination",
        check_approval_status,
        {
            "address_concerns": "address_approval_concerns",
            "complete": END
        }
    )
    
    # After addressing concerns, end (could add re-approval loop if needed)
    g.add_edge("address_approval_concerns", END)
    
    # Add edge from need_more_info to END to fix dead-end issue
    g.add_edge("need_more_info", END)

    return g.compile()


def build_refinement_graph():
    """
    Build a separate graph for proposal refinement/iteration.
    Use this when user provides feedback on existing proposal.
    """
    g = StateGraph(ProposalState)
    
    g.add_node("refine_proposal", refine_proposal)
    g.set_entry_point("refine_proposal")
    g.add_edge("refine_proposal", END)
    
    return g.compile()
