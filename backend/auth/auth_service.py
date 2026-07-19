import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

import bcrypt

from db.database import User, SessionLocal


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def generate_token() -> str:
    return secrets.token_hex(32)


class AuthService:
    _tokens = {}

    @staticmethod
    def register(username: str, password: str, role: str = "user") -> Tuple[bool, str]:
        db = SessionLocal()
        try:
            existing = db.query(User).filter(User.username == username).first()
            if existing:
                return False, "用户名已存在"

            user = User(
                username=username,
                password_hash=hash_password(password),
                role=role,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return True, "注册成功"
        except Exception as e:
            db.rollback()
            return False, f"注册失败: {str(e)}"
        finally:
            db.close()

    @staticmethod
    def login(username: str, password: str) -> Tuple[bool, str, Optional[dict]]:
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                return False, "用户不存在", None

            if not verify_password(password, user.password_hash):
                return False, "密码错误", None

            if not user.is_active:
                return False, "账户已被禁用", None

            token = generate_token()
            AuthService._tokens[token] = {
                "user_id": user.id,
                "username": user.username,
                "role": user.role,
                "expires_at": datetime.utcnow() + timedelta(days=7),
            }

            user_info = {
                "id": user.id,
                "username": user.username,
                "role": user.role,
                "token": token,
            }
            return True, "登录成功", user_info
        except Exception as e:
            return False, f"登录失败: {str(e)}", None
        finally:
            db.close()

    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        if not token:
            return None

        token_data = AuthService._tokens.get(token)
        if not token_data:
            return None

        if token_data["expires_at"] < datetime.utcnow():
            del AuthService._tokens[token]
            return None

        return token_data

    @staticmethod
    def logout(token: str) -> bool:
        if token in AuthService._tokens:
            del AuthService._tokens[token]
            return True
        return False

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[User]:
        db = SessionLocal()
        try:
            return db.query(User).filter(User.id == user_id).first()
        finally:
            db.close()

    @staticmethod
    def change_password(user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False, "用户不存在"

            if not verify_password(old_password, user.password_hash):
                return False, "原密码错误"

            user.password_hash = hash_password(new_password)
            user.updated_at = datetime.utcnow()
            db.commit()
            return True, "密码修改成功"
        except Exception as e:
            db.rollback()
            return False, f"修改失败: {str(e)}"
        finally:
            db.close()