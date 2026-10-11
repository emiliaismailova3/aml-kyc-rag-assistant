"""The structured shape of an invoice and the business rules it must satisfy.

Pydantic checks types (a date is a date, an amount is a Decimal) and the
rules below. When a rule fails, the error text is sent back to the LLM so it
can correct itself (see extract.py).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Currency = Literal["AZN", "USD", "EUR"]

# Allowed difference between the sum of line items and the stated total
# (rounding on printed invoices).
TOTAL_TOLERANCE = Decimal("0.02")


class LineItem(BaseModel):
    description: str
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    amount: Decimal = Field(ge=0)


class InvoiceData(BaseModel):
    company_name: str = Field(min_length=2)
    voen: str
    invoice_number: str = Field(min_length=1)
    issue_date: date
    total_amount: Decimal
    currency: Currency
    line_items: list[LineItem] = []

    @field_validator("voen")
    @classmethod
    def voen_is_10_digits(cls, value: str) -> str:
        value = value.strip().replace(" ", "")
        if not (value.isdigit() and len(value) == 10):
            raise ValueError("voen must be exactly 10 digits")
        return value

    @field_validator("total_amount")
    @classmethod
    def amount_is_positive(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("total_amount must be greater than 0")
        return value

    @field_validator("issue_date")
    @classmethod
    def date_not_in_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError(f"issue_date {value} is in the future")
        return value

    @model_validator(mode="after")
    def line_items_match_total(self) -> InvoiceData:
        if self.line_items:
            items_sum = sum((item.amount for item in self.line_items), Decimal(0))
            if abs(items_sum - self.total_amount) > TOTAL_TOLERANCE:
                raise ValueError(
                    f"line items sum to {items_sum} but total_amount is {self.total_amount}"
                )
        return self
