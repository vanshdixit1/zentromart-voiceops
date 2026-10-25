from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


class StoreData:
    def __init__(self, root: Path):
        self.root = root
        self.customers = self._index("customers.csv", "customer_id")
        self.orders = self._index("orders.csv", "order_id")
        self.items = read_csv(root / "data" / "order_items.csv")
        self.products = self._index("products.csv", "sku")
        self.pincodes = self._index("pincodes.csv", "pincode")
        self.holidays = {r.get("date", "") for r in read_csv(root / "data" / "holidays.csv")}

    def _index(self, filename: str, key: str) -> dict[str, dict[str, str]]:
        return {r[key]: r for r in read_csv(self.root / "data" / filename) if r.get(key)}

    def verified_order(self, order_id: str, *, phone_last4: str = "", pincode: str = "") -> dict | None:
        order = self.orders.get(order_id)
        if not order:
            return None
        customer = self.customers.get(order.get("customer_id", ""), {})
        phone = customer.get("phone", "").strip()
        supplied_phone = "".join(c for c in phone_last4 if c.isdigit())[-4:]
        supplied_pin = "".join(c for c in pincode if c.isdigit())
        verified = bool((supplied_phone and phone[-4:] == supplied_phone) or
                        (supplied_pin and order.get("delivery_pincode", "") == supplied_pin))
        if not verified:
            return {"verified": False, "message": "For privacy, please confirm the last four digits of the phone number on the order or its delivery pincode."}
        items = []
        for item in self.items:
            if item.get("order_id") == order_id:
                product = self.products.get(item.get("sku", ""), {})
                items.append({"sku": item.get("sku", ""), "product": product.get("name", ""),
                              "quantity": item.get("qty", ""), "unit_price_inr": item.get("unit_price_inr", "")})
        public_fields = {"order_id", "status", "promised_date", "shipped_date", "delivered_date",
                         "payment_status", "payment_method", "return_status", "refund_status",
                         "refund_amount_inr", "cancellation_status", "carrier", "tracking_status"}
        safe_order = {key: value for key, value in order.items() if key in public_fields and value}
        return {"verified": True, "order": safe_order, "items": items}


def add_business_days(start_iso: str, days: int, holidays: set[str]) -> str:
    from datetime import date, timedelta
    day = date.fromisoformat(start_iso)
    count = 0
    while count < days:
        day += timedelta(days=1)
        # Dataset business days are Monday through Saturday; Sunday is excluded.
        if day.weekday() != 6 and day.isoformat() not in holidays:
            count += 1
    return day.isoformat()


def refund_date_context(question: str, holidays: set[str]) -> str:
    """Deterministically calculate the published refund-arrival window when dates are explicit."""
    import re
    from datetime import date, timedelta
    text = question.casefold()
    method = None
    if "cod" in text and ("bank transfer" in text or "neft" in text):
        method = (5, 7, "COD bank transfer")
    elif "net bank" in text or "netbank" in text:
        method = (3, 5, "net banking/bank transfer")
    elif "neft" in text or "bank transfer" in text:
        method = (5, 7, "bank transfer")
    elif "upi" in text:
        method = (1, 3, "UPI")
    elif "card" in text:
        method = (5, 7, "card")
    if not method or not any(word in text for word in ("refund", "रिफंड")):
        return ""
    months = {name.casefold(): index for index, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), 1)}
    iso_match = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    named_match = re.search(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b", text, re.I)
    if iso_match:
        start = date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
    elif named_match:
        start = date(int(named_match.group(3)), months[named_match.group(2).casefold()], int(named_match.group(1)))
    else:
        return ""
    earliest, latest, label = method
    first_date = add_business_days(start.isoformat(), earliest, holidays)
    last_date = add_business_days(start.isoformat(), latest, holidays)
    today = date(2026, 10, 20)
    elapsed = 0
    cursor = date.fromisoformat(last_date)
    while cursor < today:
        cursor += timedelta(days=1)
        if cursor.weekday() != 6 and cursor.isoformat() not in holidays:
            elapsed += 1
    overdue = (f" As of 20 October 2026, the maximum date is {elapsed} business days past due; "
               "the escalation threshold is more than 3 business days." if last_date < today.isoformat() else "")
    return (f"Deterministic calendar check for the explicit refund-start date: {label} takes "
            f"{earliest} to {latest} business days after initiation. With Monday-Saturday business days, "
            f"Sundays and listed holidays skipped, the arrival window is "
            f"{first_date} through {last_date}. State the latest date clearly; "
            f"this is an estimate window, not a guarantee.{overdue}")
