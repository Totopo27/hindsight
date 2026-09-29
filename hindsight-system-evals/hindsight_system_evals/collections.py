"""One story: the folder a procurement team actually has, and the table they want out of it.

Every vendor in this space sells the same picture — a spreadsheet over a document
set, a column per question, a citation under every cell. Hebbia's Matrix puts a
row per document or entity and an agent in each cell; V7 Go sells it as data-room
and credit-agreement diligence; LlamaExtract sells a schema plus citations and
reasoning per field. The demo is always the same shape: clean documents, one fact
each, extracted once.

A procurement folder is not that, and the four things that make it not that are
what this suite grades:

* **Identity.** The same vendor is "Acme Ltd" in the MSA, "ACME Limited" in the
  invoice and "Acme" in a meeting note. One row, or the table is worse than the
  folder.
* **Supersession.** An amendment raises a contract's value. The row has to say
  what is true now and cite the amendment, not the original it replaced.
* **Absence.** A vendor nobody wrote a country for has no country. An invented
  one is the expensive failure: a wrong answer can be asked again, a wrong row is
  read back as fact.
* **Change.** A document gets corrected, another gets deleted. A table derived
  once and never rechecked is a snapshot; the row has to follow its sources.

The corpus is small on purpose — thirteen short documents — because none of the
above needs volume to break, and a suite that costs a model an hour per run does
not get run. Values are chosen so no two are confusable: every number, date and
country appears for exactly one subject, so a mis-attribution is visible rather
than plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The bank's two collections. `contracts.vendor` is a relationship: its value is
#: a vendor record's id, which is what makes the join in `col-join` possible.
VENDORS = {
    "name": "Vendors",
    "description": "A company we buy from",
    "identity": "name",
    "fields": {
        "name": {"type": "string", "description": "The vendor's name as the documents write it"},
        "country": {"type": "string", "description": "Where the vendor is registered, if a document says"},
        "tier": {"type": "string", "values": ["strategic", "standard"], "description": "Its supplier tier"},
    },
}

CONTRACTS = {
    "name": "Contracts",
    "description": "One agreement with one vendor",
    "identity": "reference",
    "fields": {
        "reference": {"type": "string", "description": "The contract reference, e.g. C-1041"},
        "value": {"type": "number", "description": "The committed annual value in EUR, as currently agreed"},
        "status": {"type": "string", "values": ["active", "expired"], "description": "Whether it is in force"},
        "signed_on": {"type": "date", "description": "The date it was signed"},
        "vendor": {"collection": "vendors", "description": "The vendor it is with"},
    },
}


@dataclass(frozen=True)
class Document:
    """One file in the folder, as plain text — the shape a converted PDF arrives in."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Case:
    """One graded question about the derived table.

    ``must_not`` is the trap: the specific wrong row the failure produces, asserted
    before correctness, exactly as the answer suites do it.
    """

    id: str
    category: str
    what: str
    must_not: str


# ---- the folder


def documents() -> list[Document]:
    """The thirteen documents, in the order they would have arrived.

    Acme is named three ways across four documents; its contract is amended once.
    Northwind is discussed without anyone ever writing down where it is registered.
    Globex is the control: one vendor, one contract, stated once, never touched.
    """
    return [
        Document(
            "msa-acme",
            "Master Services Agreement — Acme Ltd",
            "MASTER SERVICES AGREEMENT\n\n"
            "This agreement, reference C-1041, is entered into on 14 January 2025 between Northbridge "
            "Manufacturing BV and Acme Ltd, a company registered in Ireland.\n\n"
            "Acme Ltd shall provide logistics coordination services. The committed annual value is "
            "EUR 120,000, invoiced quarterly. The agreement is in force until terminated in writing.",
        ),
        Document(
            "amendment-acme-2",
            "Amendment No. 2 to C-1041",
            "AMENDMENT No. 2 to agreement C-1041 (Acme Ltd)\n\n"
            "Effective 1 September 2025, clause 4.1 is replaced. The committed annual value is increased "
            "from EUR 120,000 to EUR 185,000 to cover the additional Rotterdam lanes.\n\n"
            "All other terms of the agreement of 14 January 2025 remain unchanged.",
        ),
        Document(
            "invoice-acme-q4",
            "Invoice 2025-Q4 — ACME Limited",
            "INVOICE\n\nFrom: ACME Limited, Dublin, Ireland\nTo: Northbridge Manufacturing BV\n"
            "Period: Q4 2025\nAgainst agreement C-1041\n\nLogistics coordination, quarterly instalment: "
            "EUR 46,250\nVAT reverse charge applies.",
        ),
        Document(
            "note-supplier-review",
            "Supplier review meeting, 3 October 2025",
            "Supplier review, 3 October 2025. Present: procurement, finance.\n\n"
            "Acme was confirmed as a strategic supplier for 2026 — the Rotterdam lanes went live without "
            "incident and the amended rate is holding.\n\n"
            "Northwind Packaging was discussed as a possible second source. Nobody has run the "
            "onboarding checks yet, so it stays a standard supplier for now. We do not have their "
            "registration details on file.",
        ),
        Document(
            "msa-globex",
            "Master Services Agreement — Globex SA",
            "MASTER SERVICES AGREEMENT\n\nReference C-2207, signed 22 March 2025 between Northbridge "
            "Manufacturing BV and Globex SA, registered in France.\n\n"
            "Globex SA shall supply packaging materials. The committed annual value is EUR 64,000. "
            "The agreement is in force until terminated in writing. Globex SA is classified as a "
            "standard supplier.",
        ),
        Document(
            "msa-initech",
            "Services Agreement — Initech GmbH",
            "SERVICES AGREEMENT\n\nReference C-3115, signed 9 June 2024 between Northbridge Manufacturing "
            "BV and Initech GmbH, registered in Germany.\n\n"
            "Initech GmbH shall provide warehouse software maintenance. The committed annual value is "
            "EUR 31,500. Initech GmbH is a standard supplier.",
        ),
        Document(
            "notice-initech-expiry",
            "Non-renewal notice — C-3115",
            "NOTICE OF NON-RENEWAL\n\nDated 1 May 2025.\n\nNorthbridge Manufacturing BV gives notice that "
            "agreement C-3115 with Initech GmbH will not be renewed and expired on 8 June 2025. No further "
            "instalments are due.",
        ),
        Document(
            "note-packaging-sourcing",
            "Packaging sourcing note",
            "Packaging sourcing, September 2025.\n\nNorthwind Packaging quoted for the secondary cartons. "
            "No agreement has been signed with Northwind Packaging and no reference has been issued. "
            "They are treated as a standard supplier until onboarding completes.",
        ),
        Document(
            "security-review-acme",
            "Security review — Acme Ltd",
            "ANNUAL SECURITY REVIEW\n\nSupplier: Acme Ltd (Ireland)\nAgreement: C-1041\nReviewed: 12 November 2025\n\n"
            "No findings above informational. Acme Ltd remains approved as a strategic supplier.",
        ),
        Document(
            "note-budget-2026",
            "Budget note, committed spend 2026",
            "Committed spend for 2026 covers the agreements in force at year end: the Acme logistics "
            "agreement at its amended rate, and the Globex packaging agreement. The Initech agreement "
            "expired in June 2025 and is not in the 2026 commitment.",
        ),
        Document(
            "invoice-globex-q1",
            "Invoice 2026-Q1 — Globex SA",
            "INVOICE\n\nFrom: Globex SA, Lyon, France\nTo: Northbridge Manufacturing BV\nPeriod: Q1 2026\n"
            "Against agreement C-2207\n\nPackaging materials, quarterly instalment: EUR 16,000.",
        ),
    ]


def correction_documents() -> list[Document]:
    """The Umbrella pair, which only the correction case uses.

    It is deliberately not in the folder above: its value is contested between the
    agreement and the correction note, so leaving it in the shared bank would make
    every total over that bank a matter of opinion rather than a gradeable number.
    """
    return [
        Document(
            "msa-umbrella",
            "Services Agreement — Umbrella Logistics SL",
            "SERVICES AGREEMENT\n\nReference C-4402, signed 2 February 2026 between Northbridge "
            "Manufacturing BV and Umbrella Logistics SL, registered in Spain.\n\n"
            "Umbrella Logistics SL shall provide overflow warehousing. The committed annual value is "
            "EUR 52,000. Umbrella Logistics SL is a standard supplier.",
        ),
        Document(
            "note-umbrella-correction",
            "Correction — C-4402 value",
            "Correction issued 20 February 2026. The value stated in the Umbrella Logistics SL agreement "
            "C-4402 was transcribed incorrectly. The committed annual value is EUR 57,500, not EUR 52,000. "
            "The signed original governs and reads EUR 57,500.",
        ),
    ]


#: The one document rewritten mid-run, to grade what a correction does to a row
#: that was already derived from its earlier text. Same id: this is an edit of a
#: document the bank already holds, not a new one.
CORRECTED_DOCUMENT = Document(
    "msa-umbrella",
    "Services Agreement — Umbrella Logistics SL (corrected)",
    "SERVICES AGREEMENT\n\nReference C-4402, signed 2 February 2026 between Northbridge Manufacturing BV "
    "and Umbrella Logistics SL, registered in Spain.\n\n"
    "Umbrella Logistics SL shall provide overflow warehousing. The committed annual value is EUR 57,500 "
    "(corrected from the EUR 52,000 originally transcribed). Umbrella Logistics SL is a standard supplier.",
)

#: Deleted mid-run: the only document that says Globex is in Lyon. Deleting it
#: must take what it contributed with it.
DELETED_DOCUMENT_ID = "invoice-globex-q1"


# ---- what the table has to say


@dataclass(frozen=True)
class Expectation:
    """The gold for one record: the fields that must be right, and the ones that must stay empty."""

    record: str
    values: dict[str, object] = field(default_factory=dict)
    empty: tuple[str, ...] = ()


#: Vendors, by the identity the collection resolves on — which is the *normalised*
#: name, so "Acme Ltd" and "ACME Limited" are both the record "acme" and a contract
#: reference "C-1041" is the record "c 1041". Acme is the identity case: three
#: spellings across four documents, one row.
VENDOR_GOLD = (
    Expectation("acme", {"country": "Ireland", "tier": "strategic"}),
    Expectation("globex", {"country": "France", "tier": "standard"}),
    Expectation("initech", {"country": "Germany", "tier": "standard"}),
    # Nobody ever wrote where Northwind is registered. An invented country here is
    # the same failure class as a fabricated fact dimension in retain.
    Expectation("northwind packaging", {"tier": "standard"}, empty=("country",)),
)

#: Contracts. C-1041 is the supersession case, C-3115 the status case.
CONTRACT_GOLD = (
    Expectation("c 1041", {"value": 185000, "status": "active", "signed_on": "2025-01-14"}),
    Expectation("c 2207", {"value": 64000, "status": "active", "signed_on": "2025-03-22"}),
    Expectation("c 3115", {"value": 31500, "status": "expired", "signed_on": "2024-06-09"}),
)


def cases() -> list[Case]:
    """The graded cases, each naming the wrong row it exists to catch."""
    return [
        Case(
            "col-identity",
            "identity",
            "Acme is written 'Acme Ltd', 'ACME Limited' and 'Acme' across four documents, and is one vendor record.",
            "two or more separate vendor records for Acme",
        ),
        Case(
            "col-supersession",
            "supersession",
            "C-1041 carries the amended value of 185,000, and its evidence cites the amendment.",
            "the superseded value of 120,000",
        ),
        Case(
            "col-absence",
            "absence",
            "Northwind Packaging has no country, because no document states one.",
            "any country at all for Northwind Packaging",
        ),
        Case(
            "col-fields",
            "accuracy",
            "Every field the folder states is in the table with the value the folder states.",
            "a field filled with a value no document gives",
        ),
        Case(
            "col-fragments",
            "identity",
            "Every row in the table is a thing: a contract row has a reference, a vendor row is a vendor we buy from.",
            "rows with no identity at all, or the buyer filed as one of its own vendors",
        ),
        Case(
            "col-join",
            "relationship",
            "Each contract's vendor field resolves to the vendor record, so a join returns the vendor's "
            "country beside the contract.",
            "a contract whose vendor does not resolve to a vendor record",
        ),
        Case(
            "col-evidence",
            "evidence",
            "Every filled field cites a document that actually contains that value.",
            "a value whose evidence points at a document that does not say it",
        ),
        Case(
            "col-status",
            "supersession",
            "C-3115 is expired, because a later notice says so, while C-1041 and C-2207 stay active.",
            "C-3115 still active",
        ),
        Case(
            "col-correction",
            "change",
            "After the Umbrella agreement is rewritten with the corrected figure, C-4402 reads 57,500.",
            "the withdrawn figure of 52,000",
        ),
        Case(
            "col-deletion",
            "change",
            "After the Globex invoice is deleted, nothing in the table still rests on it.",
            "a value still citing the deleted invoice",
        ),
        Case(
            "col-aggregate",
            "query",
            "Committed annual value across contracts in force is 249,000 (Acme 185,000 + Globex 64,000).",
            "a total that counts the expired contract or the superseded value",
        ),
    ]


#: The cases that are the reason this suite exists — one row per real vendor, the
#: amendment beating the original, and nothing invented. The rest run with --full.
MINIMUM_ACCEPTANCE = ("col-identity", "col-supersession", "col-absence")
