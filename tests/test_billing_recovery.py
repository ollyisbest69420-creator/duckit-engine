"""Regression checks for the Duck It billing recovery paths.

Run with the repo root on ``sys.path`` (the test adds it itself):

    python tests/test_billing_recovery.py

The checks exercise the helpers added so a Render redeploy (which wipes the
SQLite volume) can rebuild Pro entitlements from Stripe, and so a Stripe outage
never downgrades a paying customer. No network or Stripe account is used; the
``stripe`` module is replaced with a small in-memory fake.
"""

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TMP = tempfile.mkdtemp(prefix="duckit-test-")
os.environ["APP_BASE_URL"] = "http://localhost:8000"
os.environ["APP_SESSION_SECRET"] = "test-secret"
os.environ["DUCKIT_DATA_DIR"] = TMP
os.environ["ENVIRONMENT"] = "development"
for key in ("STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "STRIPE_PRICE_PRO_MONTHLY"):
    os.environ.pop(key, None)

import server  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import Response  # noqa: E402

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print("PASS", label)
    else:
        FAILURES.append(label)
        print("FAIL", label, detail)


class ListObj:
    def __init__(self, data):
        self.data = data


def make_sub(sid, customer, status, identity=None, created=100):
    metadata = {"duckit_identity_id": identity} if identity else {}
    return {"id": sid, "customer": customer, "status": status, "created": created, "metadata": metadata}


class FakeStripe:
    """Minimal stand-in for the bits of the Stripe SDK the server touches."""

    def __init__(self, subs_by_customer=None, search_results=None, subs_by_id=None, customers_by_id=None):
        outer = self
        self.subs_by_customer = subs_by_customer or {}
        self.search_results = search_results or []
        self.subs_by_id = subs_by_id or {}
        self.customers_by_id = customers_by_id or {}
        self.fail = False
        self.list_calls = []
        self.search_queries = []

        class Subscription:
            @staticmethod
            def list(customer=None, status=None, limit=None):
                if outer.fail:
                    raise RuntimeError("stripe is down")
                outer.list_calls.append(customer)
                return ListObj(list(outer.subs_by_customer.get(customer, [])))

            @staticmethod
            def search(query=None, limit=None):
                if outer.fail:
                    raise RuntimeError("stripe is down")
                outer.search_queries.append(query)
                return ListObj(list(outer.search_results))

            @staticmethod
            def retrieve(subscription_id):
                if outer.fail:
                    raise RuntimeError("stripe is down")
                return outer.subs_by_id[subscription_id]

        class Customer:
            @staticmethod
            def list(email=None, limit=None):
                if outer.fail:
                    raise RuntimeError("stripe is down")
                return ListObj([])

            @staticmethod
            def retrieve(customer_id):
                if outer.fail:
                    raise RuntimeError("stripe is down")
                return outer.customers_by_id[customer_id]

        self.Subscription = Subscription
        self.Customer = Customer


def use_fake(fake):
    server.stripe = fake
    server.STRIPE_SECRET_KEY = "sk_test_fake"


def make_request(cookie_value=None):
    headers = []
    if cookie_value:
        headers.append((b"cookie", f"duckit_identity={cookie_value}".encode()))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})


def cookie_value_from(response):
    header = response.headers.get("set-cookie")
    if not header:
        return None
    return header.split("duckit_identity=", 1)[1].split(";", 1)[0]


def wipe_rows():
    """Emulate a deploy that replaces the SQLite file."""
    with server.db() as conn:
        conn.execute("DELETE FROM identities")
        conn.execute("DELETE FROM checkout_sessions")


def reset_clocks():
    for clock in (server.recovery_pending, server.recovery_attempts, server.revalidated_at, server.reconcile_attempts):
        clock.clear()


def run(label, func):
    try:
        func()
    except Exception:
        FAILURES.append(label)
        print("FAIL", label, "(exception)")
        traceback.print_exc()


def test_helpers():
    check("identity id accepts 32 hex", server.is_identity_id("a" * 32))
    check(
        "identity id rejects uppercase/short/None",
        not server.is_identity_id("A" * 32) and not server.is_identity_id("a" * 31) and not server.is_identity_id(None),
    )
    check("stripe search escaping", server.stripe_search_escape("a'b\\c") == "a\\'b\\\\c")
    now = int(time.time())
    fresh = {"subscription_status": "none", "subscription_id": None, "created_at": now - 3600}
    check("fresh row is adoptable", server.identity_is_fresh(fresh))
    check("old row is not adoptable", not server.identity_is_fresh(dict(fresh, created_at=now - 3 * 24 * 3600)))
    check("billing history blocks adoption", not server.identity_is_fresh(dict(fresh, subscription_status="active")))
    check("missing row is not adoptable", not server.identity_is_fresh(None))


def test_link_conflict():
    wipe_rows()
    owner, other = "1" * 32, "2" * 32
    server.ensure_identity_row(owner, customer_id="cus_owner")
    server.ensure_identity_row(other)
    with server.db() as conn:
        server.link_billing_ids(conn, other, customer_id="cus_owner")
    rows = {}
    with server.db() as conn:
        for row in conn.execute("SELECT * FROM identities").fetchall():
            rows[row["id"]] = dict(row)
    check("owner keeps the customer", rows[owner]["stripe_customer_id"] == "cus_owner", rows[owner])
    check("second row does not steal it", rows[other]["stripe_customer_id"] is None, rows[other])


def all_rows():
    rows = {}
    with server.db() as conn:
        for row in conn.execute("SELECT * FROM identities").fetchall():
            rows[row["id"]] = dict(row)
    return rows


def test_save_subscription():
    wipe_rows()
    identity = "3" * 32
    fake = FakeStripe()
    use_fake(fake)
    saved = server.save_subscription(make_sub("sub_a", "cus_a", "active", identity=identity), identity)
    row = server.user_row(identity)
    check("save with hint returns identity", saved == identity, saved)
    check("save stores subscription and customer", row["subscription_id"] == "sub_a" and row["stripe_customer_id"] == "cus_a", dict(row))
    check("save marks pro", server.is_pro(row))

    # A second identity must not blow up when the customer sits on another row.
    second = "4" * 32
    server.ensure_identity_row(second)
    saved = server.save_subscription(make_sub("sub_b", "cus_a", "active", identity=second), second)
    rows = all_rows()
    check("duplicate customer does not raise", saved == second, saved)
    check("duplicate customer stays on owner", rows[identity]["stripe_customer_id"] == "cus_a", rows[identity])
    check("second identity keeps its own subscription", rows[second]["subscription_id"] == "sub_b", rows[second])

    # Moving a subscription between rows must not trip the UNIQUE index either.
    moved = server.save_subscription(make_sub("sub_b", "cus_b", "active", identity=identity), identity)
    rows = all_rows()
    check("subscription moved to the hinted identity", moved == identity and rows[identity]["subscription_id"] == "sub_b", rows.get(identity))
    check("previous holder loses the subscription", rows[second]["subscription_id"] is None and not server.is_pro(rows[second]), rows[second])

    # Webhooks arrive without a hint: resolve through the local row first.
    wipe_rows()
    owner = "5" * 32
    server.ensure_identity_row(owner, customer_id="cus_local")
    fake.customers_by_id = {"cus_local": {"id": "cus_local", "metadata": {"duckit_identity_id": "9" * 32}}}
    resolved = server.save_subscription(make_sub("sub_local", "cus_local", "active"))
    check("webhook resolves the local row", resolved == owner, resolved)

    # Nothing local: fall back to the identity stored on the Stripe customer.
    wiped_customer_identity = "6" * 32
    fake.customers_by_id = {"cus_meta": {"id": "cus_meta", "metadata": {"duckit_identity_id": wiped_customer_identity}}}
    resolved = server.save_subscription(make_sub("sub_meta", "cus_meta", "active"))
    check("webhook rebuilds a wiped row from metadata", resolved == wiped_customer_identity, resolved)
    check("rebuilt row is pro", server.is_pro(server.user_row(wiped_customer_identity)))

    # Unknown payer: log and carry on instead of raising inside the webhook.
    fake.customers_by_id = {}
    check("unknown payer returns None", server.save_subscription(make_sub("sub_unknown", "cus_unknown", "active")) is None)


def test_find_subscription():
    wipe_rows()
    identity = "7" * 32
    canceled = make_sub("sub_old", "cus_x", "canceled", identity=identity, created=500)
    active = make_sub("sub_new", "cus_x", "active", identity=identity, created=400)

    fake = FakeStripe(subs_by_customer={"cus_x": [canceled, active]})
    use_fake(fake)
    lookup = server.find_stripe_subscription(identity, customer_id="cus_x")
    check("picks the active subscription", lookup.checked and lookup.subscription["id"] == "sub_new", lookup.subscription)
    check("returns the customer id", lookup.customer_id == "cus_x", lookup.customer_id)

    mine = make_sub("sub_mine", "cus_y", "active", identity=identity, created=100)
    theirs = make_sub("sub_theirs", "cus_y", "active", identity="8" * 32, created=900)
    fake = FakeStripe(subs_by_customer={"cus_y": [theirs, mine]})
    use_fake(fake)
    lookup = server.find_stripe_subscription(identity, customer_id="cus_y")
    check("identity metadata breaks ties", lookup.subscription["id"] == "sub_mine", lookup.subscription)

    fake = FakeStripe(search_results=[canceled, active])
    use_fake(fake)
    lookup = server.find_stripe_subscription(identity)
    check("falls back to metadata search", lookup.checked and lookup.subscription["id"] == "sub_new", lookup.subscription)
    check("search query targets the identity", fake.search_queries and identity in fake.search_queries[0], fake.search_queries)

    fake = FakeStripe()
    use_fake(fake)
    lookup = server.find_stripe_subscription(identity, customer_id="cus_empty")
    check("no subscription is a checked miss", lookup.checked and lookup.subscription is None, lookup)

    fake = FakeStripe()
    fake.fail = True
    use_fake(fake)
    lookup = server.find_stripe_subscription(identity, customer_id="cus_down")
    check("stripe outage is unchecked", lookup.checked is False and lookup.subscription is None, lookup)


def test_apply_lookup():
    wipe_rows()
    identity = "a1" * 16
    server.ensure_identity_row(identity)
    with server.db() as conn:
        conn.execute("UPDATE identities SET subscription_status = 'active' WHERE id = ?", (identity,))
    payload = server.apply_stripe_lookup(identity, server.StripeSubscriptionLookup(True, None, None))
    check("checked miss clears the stale status", payload["is_pro"] is False and payload["subscription_status"] == "none", payload)
    with server.db() as conn:
        conn.execute("UPDATE identities SET subscription_status = 'active' WHERE id = ?", (identity,))
    payload = server.apply_stripe_lookup(identity, server.StripeSubscriptionLookup(False, None, None))
    check("outage keeps the stored entitlement", payload["is_pro"] is True, payload)


def test_clocks():
    reset_clocks()
    identity = "b2" * 16
    check("recovery needs a pending mark", not server.should_attempt_recovery(identity))
    server.mark_recovery_pending(identity)
    check("pending recovery is attempted once", server.should_attempt_recovery(identity))
    check("pending recovery is throttled", not server.should_attempt_recovery(identity))
    server.finish_recovery(identity, True)
    check("settled recovery stops looking", not server.should_attempt_recovery(identity))
    other = "c3" * 16
    server.mark_recovery_pending(other)
    server.should_attempt_recovery(other)
    server.finish_recovery(other, False)
    check("unchecked recovery stays pending", other in server.recovery_pending and not server.should_attempt_recovery(other))
    check("revalidation is throttled", server.should_revalidate_pro(other) and not server.should_revalidate_pro(other))


def test_current_identity_recovery():
    wipe_rows()
    reset_clocks()
    fake = FakeStripe()
    use_fake(fake)
    response = Response()
    identity = server.current_identity(make_request(), response)
    signed = cookie_value_from(response)
    check("mints an identity cookie", bool(signed) and server.is_identity_id(identity), (identity, signed))

    # Simulate a Render deploy: the signed cookie survives, the SQLite row does not.
    wipe_rows()
    reset_clocks()
    paid = make_sub("sub_1", "cus_1", "active", identity=identity)
    fake.search_results = [paid]
    fake.subs_by_id = {"sub_1": paid}
    fake.customers_by_id = {"cus_1": {"id": "cus_1", "metadata": {"duckit_identity_id": identity}}}
    response = Response()
    restored = server.current_identity(make_request(signed), response)
    row = server.user_row(identity)
    check("identity survives the wipe", restored == identity, restored)
    check("pro entitlement is rebuilt", server.is_pro(row), dict(row) if row else None)
    check("stripe customer is relinked", bool(row) and row["stripe_customer_id"] == "cus_1", dict(row) if row else None)
    check("recovery settles once checked", identity not in server.recovery_pending, server.recovery_pending)
    new_cookie = cookie_value_from(response)
    decoded = server.signer.loads(new_cookie, max_age=server.COOKIE_MAX_AGE) if new_cookie else None
    check("cookie is refreshed with the customer", isinstance(decoded, dict) and decoded.get("customer_id") == "cus_1", decoded)


def test_status_recovery():
    wipe_rows()
    reset_clocks()
    identity = "d4" * 16
    server.ensure_identity_row(identity)
    server.mark_recovery_pending(identity)
    paid = make_sub("sub_2", "cus_2", "active", identity=identity)
    fake = FakeStripe(search_results=[paid], subs_by_id={"sub_2": paid})
    use_fake(fake)
    payload = server.billing_status(make_request(), Response(), None, identity)
    check("status endpoint recovers pro", payload["pro"] is True, payload)
    check("status flags stay quiet", payload["reconciling"] is False and payload["pending_checkout"] is False, payload)

    # Without a pending mark the status endpoint must not call Stripe.
    wipe_rows()
    reset_clocks()
    free_identity = "e5" * 16
    server.ensure_identity_row(free_identity)
    called_before = len(fake.search_queries)
    payload = server.billing_status(make_request(), Response(), None, free_identity)
    check("idle free identity makes no stripe call", len(fake.search_queries) == called_before, fake.search_queries)
    check("idle free identity stays free", payload["pro"] is False, payload)


def test_pro_lapse_and_outage():
    wipe_rows()
    reset_clocks()
    identity = "f6" * 16
    active = make_sub("sub_4", "cus_4", "active", identity=identity)
    fake = FakeStripe(subs_by_customer={"cus_4": [active]}, subs_by_id={"sub_4": active})
    use_fake(fake)
    server.sync_pro_status_from_stripe(identity, customer_id="cus_4")
    check("initial sync grants pro", server.is_pro(server.user_row(identity)), dict(server.user_row(identity)))

    canceled = make_sub("sub_4", "cus_4", "canceled", identity=identity)
    fake.subs_by_customer = {"cus_4": [canceled]}
    fake.subs_by_id = {"sub_4": canceled}
    server.revalidated_at.clear()
    payload = server.billing_status(make_request(), Response(), None, identity)
    check("canceled subscription downgrades", payload["pro"] is False, payload)
    check("downgrade is persisted", server.user_row(identity)["subscription_status"] == "canceled", dict(server.user_row(identity)))

    wipe_rows()
    reset_clocks()
    outage_identity = "aa" * 16
    active = make_sub("sub_5", "cus_5", "active", identity=outage_identity)
    fake = FakeStripe(subs_by_customer={"cus_5": [active]}, subs_by_id={"sub_5": active})
    use_fake(fake)
    server.sync_pro_status_from_stripe(outage_identity, customer_id="cus_5")
    check("outage baseline is pro", server.is_pro(server.user_row(outage_identity)))
    fake.fail = True
    server.revalidated_at.clear()
    payload = server.billing_status(make_request(), Response(), None, outage_identity)
    check("stripe outage keeps pro", payload["pro"] is True, payload)


def main():
    run("helpers", test_helpers)
    run("link conflict", test_link_conflict)
    run("save subscription", test_save_subscription)
    run("find subscription", test_find_subscription)
    run("apply lookup", test_apply_lookup)
    run("clocks", test_clocks)
    run("current identity recovery", test_current_identity_recovery)
    run("status recovery", test_status_recovery)
    run("pro lapse and outage", test_pro_lapse_and_outage)

    print()
    if FAILURES:
        print("FAILURES:", FAILURES)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
