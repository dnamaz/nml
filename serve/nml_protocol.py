"""
NML Agent-to-Agent Communication Protocol

Message envelope and protocol definitions for structured communication
between NML agents (drafter, validator, executor, explainer).
"""

import uuid
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Message type constants
# ---------------------------------------------------------------------------

MSG_DRAFT_NML = "draft_nml"
MSG_VALIDATED_NML = "validated_nml"
MSG_TEST_REQUEST = "test_request"
MSG_EXECUTION_RESULT = "execution_result"
MSG_EXPLANATION = "explanation"

VALIDATION_DRAFT = "draft"
VALIDATION_GRAMMAR = "grammar_valid"
VALIDATION_SEMANTIC = "semantically_valid"
VALIDATION_TESTED = "tested"
VALIDATION_PRODUCTION = "production"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MessageHeader:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    source_agent: str = ""
    target_agent: str = ""
    message_type: str = ""
    priority: int = 0  # 0=routine … 9=critical


@dataclass
class Provenance:
    source_document: str = ""
    section_ref: str = ""
    confidence: float = 1.0
    reasoning_hash: str = ""


@dataclass
class NMLPayload:
    program: str = ""
    syntax_variant: str = "symbolic"  # symbolic | classic | verbose
    validation_status: str = VALIDATION_DRAFT
    instruction_count: int = 0
    fragment_name: str = ""
    is_patch: bool = False


@dataclass
class MessageContext:
    jurisdiction_key: str = ""
    tax_year: int = 0
    effective_date: str = ""
    prior_version_hash: str = ""


@dataclass
class ProgramDescriptor:
    name: str = ""
    version: str = ""
    domain: str = ""
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    invariants: list = field(default_factory=list)
    provenance: str = ""
    author: str = ""
    created: str = ""


@dataclass
class AgentMessage:
    header: MessageHeader = field(default_factory=MessageHeader)
    provenance: Provenance = field(default_factory=Provenance)
    payload: NMLPayload = field(default_factory=NMLPayload)
    context: MessageContext = field(default_factory=MessageContext)
    descriptor: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data) -> "AgentMessage":
        if isinstance(data, str):
            data = json.loads(data)
        return cls(
            header=MessageHeader(**data.get("header", {})),
            provenance=Provenance(**data.get("provenance", {})),
            payload=NMLPayload(**data.get("payload", {})),
            context=MessageContext(**data.get("context", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMessage":
        return cls.from_json(data)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def compute_nml_hash(nml_program: str) -> str:
    """Return the SHA-256 hex digest of an NML program string."""
    return hashlib.sha256(nml_program.encode("utf-8")).hexdigest()


def extract_descriptor(nml_program: str) -> ProgramDescriptor:
    """Parse META lines from an NML program and build a ProgramDescriptor."""
    desc = ProgramDescriptor()
    for line in nml_program.split("\n"):
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if not tokens:
            continue
        op = tokens[0]
        if op not in ("META", "§", "METADATA"):
            continue
        if len(tokens) < 3:
            continue
        key = tokens[1].lstrip("@")
        value = " ".join(tokens[2:]).strip('"')

        if key == "name":
            desc.name = value
        elif key == "version":
            desc.version = value
        elif key == "domain":
            desc.domain = value
        elif key == "provenance":
            desc.provenance = value
        elif key == "author":
            desc.author = value
        elif key == "created":
            desc.created = value
        elif key == "input":
            parts = tokens[2:]
            entry = {"name": parts[0] if parts else "", "type": parts[1] if len(parts) > 1 else "float"}
            if len(parts) > 2:
                entry["description"] = " ".join(parts[2:]).strip('"')
            desc.inputs.append(entry)
        elif key == "output":
            parts = tokens[2:]
            entry = {"name": parts[0] if parts else "", "type": parts[1] if len(parts) > 1 else "float"}
            if len(parts) > 2:
                entry["description"] = " ".join(parts[2:]).strip('"')
            desc.outputs.append(entry)
        elif key == "invariant":
            desc.invariants.append(value)
    return desc


def create_draft_message(
    source_agent: str,
    target_agent: str,
    nml_program: str,
    jurisdiction_key: str,
    tax_year: int,
    syntax: str = "symbolic",
    source_document: str = "",
    confidence: float = 1.0,
) -> AgentMessage:
    """Convenience constructor for a draft NML message."""
    program_hash = compute_nml_hash(nml_program)
    instruction_count = sum(
        1 for line in nml_program.splitlines()
        if line.strip() and not line.strip().startswith("//")
    )

    return AgentMessage(
        header=MessageHeader(
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=MSG_DRAFT_NML,
        ),
        provenance=Provenance(
            source_document=source_document,
            confidence=confidence,
            reasoning_hash=program_hash,
        ),
        payload=NMLPayload(
            program=nml_program,
            syntax_variant=syntax,
            validation_status=VALIDATION_DRAFT,
            instruction_count=instruction_count,
        ),
        context=MessageContext(
            jurisdiction_key=jurisdiction_key,
            tax_year=tax_year,
            prior_version_hash="",
        ),
    )


def create_validation_message(
    source_msg: AgentMessage,
    validation_status: str,
    source_agent: str = "validator",
) -> AgentMessage:
    """Create a validation response, copying context from *source_msg*."""
    return AgentMessage(
        header=MessageHeader(
            source_agent=source_agent,
            target_agent=source_msg.header.source_agent,
            message_type=MSG_VALIDATED_NML,
        ),
        provenance=Provenance(
            source_document=source_msg.provenance.source_document,
            confidence=source_msg.provenance.confidence,
            reasoning_hash=source_msg.provenance.reasoning_hash,
        ),
        payload=NMLPayload(
            program=source_msg.payload.program,
            syntax_variant=source_msg.payload.syntax_variant,
            validation_status=validation_status,
            instruction_count=source_msg.payload.instruction_count,
        ),
        context=MessageContext(
            jurisdiction_key=source_msg.context.jurisdiction_key,
            tax_year=source_msg.context.tax_year,
            effective_date=source_msg.context.effective_date,
            prior_version_hash=compute_nml_hash(source_msg.payload.program),
        ),
    )


def create_execution_message(
    source_msg: AgentMessage,
    outputs: dict,
    cycles: int,
    time_us: float,
    source_agent: str = "engine",
) -> AgentMessage:
    """Create an execution-result message carrying engine outputs."""
    result_payload = json.dumps(
        {"outputs": outputs, "cycles": cycles, "time_us": time_us},
        indent=2,
    )

    return AgentMessage(
        header=MessageHeader(
            source_agent=source_agent,
            target_agent=source_msg.header.source_agent,
            message_type=MSG_EXECUTION_RESULT,
        ),
        provenance=Provenance(
            source_document=source_msg.provenance.source_document,
            confidence=source_msg.provenance.confidence,
            reasoning_hash=compute_nml_hash(result_payload),
        ),
        payload=NMLPayload(
            program=result_payload,
            syntax_variant=source_msg.payload.syntax_variant,
            validation_status=source_msg.payload.validation_status,
            instruction_count=source_msg.payload.instruction_count,
        ),
        context=MessageContext(
            jurisdiction_key=source_msg.context.jurisdiction_key,
            tax_year=source_msg.context.tax_year,
            effective_date=source_msg.context.effective_date,
            prior_version_hash=compute_nml_hash(source_msg.payload.program),
        ),
    )


def create_described_message(
    source_agent: str,
    target_agent: str,
    nml_program: str,
    jurisdiction_key: str,
    tax_year: int,
    syntax: str = "symbolic",
    source_document: str = "",
    confidence: float = 1.0,
) -> AgentMessage:
    """Like create_draft_message but also extracts and includes the program descriptor."""
    msg = create_draft_message(
        source_agent, target_agent, nml_program,
        jurisdiction_key, tax_year, syntax, source_document, confidence,
    )
    desc = extract_descriptor(nml_program)
    if desc.provenance:
        msg.provenance.source_document = desc.provenance
    return msg
