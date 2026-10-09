"""Authentication, Password Hashing, and Cryptographic HMAC Token Services."""

import base64
import hashlib
import hmac
import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

JWT_SECRET = os.getenv("JWT_SECRET", "syntra_openpilot_secret_key_2026_super_safe_token")
JWT_EXPIRATION_SECONDS = 86400 * 7  # 7 days

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email(email: str) -> bool:
    """Validates email format using RFC compliant pattern."""
    return bool(EMAIL_REGEX.match(email.strip()))


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validates that password meets security requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(c.isdigit() for c in password) and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        return False, "Password must contain at least one number or special character."
    return True, ""


def hash_password(password: str) -> Tuple[str, str]:
    """
    Hashes password using PBKDF2-HMAC-SHA256 with 100,000 iterations and a random salt.
    Returns (hex_hash, hex_salt).
    """
    salt = os.urandom(16).hex()
    pwd_bytes = password.encode("utf-8")
    salt_bytes = salt.encode("utf-8")
    hashed = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100_000).hex()
    return hashed, salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verifies a plaintext password against stored hash and salt using constant-time comparison."""
    pwd_bytes = password.encode("utf-8")
    salt_bytes = salt.encode("utf-8")
    computed = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100_000).hex()
    return hmac.compare_digest(computed, password_hash)


def create_access_token(user_id: str, email: str, is_demo: bool = False) -> str:
    """
    Generates a cryptographically signed URL-safe token with tamper-proof HMAC-SHA256 signature.
    """
    now = int(time.time())
    payload = {
        "sub": user_id,
        "email": email,
        "is_demo": is_demo,
        "iat": now,
        "exp": now + JWT_EXPIRATION_SECONDS
    }
    payload_bytes = json.dumps(payload, separators=(',', ':')).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and verifies token HMAC-SHA256 signature and expiration timestamp.
    Returns payload dictionary or None if invalid or expired.
    """
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None

        # Re-pad base64 string
        padding = "=" * ((4 - len(payload_b64) % 4) % 4)
        raw_json = base64.urlsafe_b64decode((payload_b64 + padding).encode("utf-8")).decode("utf-8")
        payload = json.loads(raw_json)

        # Verify expiration
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except Exception:
        return None
