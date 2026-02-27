import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from cryptography.fernet import Fernet
import bcrypt
import json
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Legacy table - kept for backward compatibility
class UserConfig(Base):
    __tablename__ = 'user_configs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, unique=True)
    db_host = Column(Text)
    db_port = Column(Text)
    db_name = Column(Text)
    db_user = Column(Text)
    db_password = Column(Text)
    openai_api_key = Column(Text)
    anthropic_api_key = Column(Text)
    gemini_api_key = Column(Text)
    grok_api_key = Column(Text)

# --- NEW: Multiple DB Connections per user ---
class DBConnection(Base):
    __tablename__ = 'db_connections'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    profile_name = Column(Text, default='Mặc định')
    db_host = Column(Text)
    db_port = Column(Text)
    db_name = Column(Text)
    db_user = Column(Text)
    db_password = Column(Text)
    is_default = Column(Boolean, default=False)

# --- NEW: Multiple API Keys per user ---
class APIKey(Base):
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    profile_name = Column(Text, default='Mặc định')
    provider = Column(Text)  # OpenAI, Grok (xAI), Gemini, Claude
    api_key = Column(Text)   # Encrypted
    is_default = Column(Boolean, default=False)

class UserDashboard(Base):
    __tablename__ = 'user_dashboards'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    data = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AppDBManager:
    def __init__(self, db_url="postgresql+psycopg://postgres:AppraisalQuail1Agent@103.118.28.2:5432/chatdb", encryption_key=None):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
        else:
            key = os.getenv("ENCRYPTION_KEY")
            if not key:
                print("WARNING: Creating a temporary encryption key. Data will be lost on restart.")
                key = Fernet.generate_key().decode()
            self.fernet = Fernet(key.encode())

    def encrypt(self, data):
        if not data:
            return None
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, data):
        if not data:
            return None
        try:
            return self.fernet.decrypt(data.encode()).decode()
        except:
            return None

    # ========================
    # User Management
    # ========================
    
    def create_user(self, username, email, password):
        if self.session.query(User).filter_by(username=username).first():
            return False, "Tên đăng nhập đã tồn tại"
        if self.session.query(User).filter_by(email=email).first():
            return False, "Email này đã được sử dụng"
            
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(
            username=username,
            email=email,
            password_hash=hashed.decode('utf-8')
        )
        self.session.add(new_user)
        self.session.commit()
        return True, new_user.id

    def reset_password_by_auth(self, username, email, new_password):
        """Reset password if both username and email match the database"""
        user = self.session.query(User).filter_by(username=username, email=email).first()
        if not user:
            return False, "Tên đăng nhập hoặc Email không khớp"
        
        if len(new_password) < 6:
            return False, "Mật khẩu mới phải có ít nhất 6 ký tự"
            
        user.password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        self.session.commit()
        return True, "Cấp lại mật khẩu thành công"

    def verify_user(self, username, password):
        user = self.session.query(User).filter_by(username=username).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            return True, user.id
        return False, None
        
    def get_user_by_username(self, username):
        return self.session.query(User).filter_by(username=username).first()

    def change_password(self, user_id, old_password, new_password):
        user = self.session.query(User).filter_by(id=user_id).first()
        if not user:
            return False, "Người dùng không tồn tại"
        if not bcrypt.checkpw(old_password.encode('utf-8'), user.password_hash.encode('utf-8')):
            return False, "Mật khẩu cũ không đúng"
        if len(new_password) < 6:
            return False, "Mật khẩu mới phải có ít nhất 6 ký tự"
        user.password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        self.session.commit()
        return True, "Đổi mật khẩu thành công"

    # ========================
    # DB Connections (Multiple)
    # ========================

    def get_db_connections(self, user_id):
        """Get all DB connections for a user, decrypted"""
        conns = self.session.query(DBConnection).filter_by(user_id=user_id).all()
        result = []
        for c in conns:
            result.append({
                'id': c.id,
                'profile_name': c.profile_name or 'Mặc định',
                'db_host': self.decrypt(c.db_host) or '',
                'db_port': self.decrypt(c.db_port) or '',
                'db_name': self.decrypt(c.db_name) or '',
                'db_user': self.decrypt(c.db_user) or '',
                'db_password': self.decrypt(c.db_password) or '',
                'is_default': c.is_default or False,
            })
        return result

    def save_db_connection(self, user_id, profile_name, db_host, db_port, db_name, db_user, db_password, is_default=False, conn_id=None):
        """Save or update a DB connection"""
        if conn_id:
            conn = self.session.query(DBConnection).filter_by(id=conn_id, user_id=user_id).first()
        else:
            conn = None
        
        if not conn:
            conn = DBConnection(user_id=user_id)
            self.session.add(conn)
        
        conn.profile_name = profile_name
        conn.db_host = self.encrypt(db_host)
        conn.db_port = self.encrypt(db_port)
        conn.db_name = self.encrypt(db_name)
        conn.db_user = self.encrypt(db_user)
        conn.db_password = self.encrypt(db_password)
        conn.is_default = is_default
        
        # If this is default, unset other defaults
        if is_default:
            self.session.query(DBConnection).filter(
                DBConnection.user_id == user_id,
                DBConnection.id != conn.id
            ).update({'is_default': False})
        
        self.session.commit()
        return conn.id

    def delete_db_connection(self, conn_id, user_id):
        """Delete a DB connection"""
        self.session.query(DBConnection).filter_by(id=conn_id, user_id=user_id).delete()
        self.session.commit()

    # ========================
    # API Keys (Multiple)
    # ========================

    def get_api_keys(self, user_id):
        """Get all API keys for a user, decrypted"""
        keys = self.session.query(APIKey).filter_by(user_id=user_id).all()
        result = []
        for k in keys:
            result.append({
                'id': k.id,
                'profile_name': k.profile_name or 'Mặc định',
                'provider': k.provider or '',
                'api_key': self.decrypt(k.api_key) or '',
                'is_default': k.is_default or False,
            })
        return result

    def save_api_key(self, user_id, profile_name, provider, api_key, is_default=False, key_id=None):
        """Save or update an API key"""
        # Check duplicate
        existing_keys = self.get_api_keys(user_id)
        for k in existing_keys:
            if k['api_key'] == api_key and str(k['id']) != str(key_id):
                return False, f"API Key này đã tồn tại với tên: {k['profile_name']}"

        if key_id:
            rec = self.session.query(APIKey).filter_by(id=key_id, user_id=user_id).first()
        else:
            rec = None
        
        if not rec:
            rec = APIKey(user_id=user_id)
            self.session.add(rec)
        
        rec.profile_name = profile_name
        rec.provider = provider
        rec.api_key = self.encrypt(api_key)
        rec.is_default = is_default
        
        # If this is default for this provider, unset other defaults
        if is_default:
            self.session.query(APIKey).filter(
                APIKey.user_id == user_id,
                APIKey.provider == provider,
                APIKey.id != rec.id
            ).update({'is_default': False})
        
        self.session.commit()
        return True, "Lưu thành công"

    def delete_api_key(self, key_id, user_id):
        """Delete an API key"""
        self.session.query(APIKey).filter_by(id=key_id, user_id=user_id).delete()
        self.session.commit()

    # ========================
    # Legacy Config (backward compat)
    # ========================

    def save_config(self, user_id, config_data):
        config = self.session.query(UserConfig).filter_by(user_id=user_id).first()
        if not config:
            config = UserConfig(user_id=user_id)
            self.session.add(config)
        if 'db_host' in config_data: config.db_host = self.encrypt(config_data['db_host'])
        if 'db_port' in config_data: config.db_port = self.encrypt(config_data['db_port'])
        if 'db_name' in config_data: config.db_name = self.encrypt(config_data['db_name'])
        if 'db_user' in config_data: config.db_user = self.encrypt(config_data['db_user'])
        if 'db_password' in config_data: config.db_password = self.encrypt(config_data['db_password'])
        if 'openai_api_key' in config_data: config.openai_api_key = self.encrypt(config_data['openai_api_key'])
        if 'anthropic_api_key' in config_data: config.anthropic_api_key = self.encrypt(config_data['anthropic_api_key'])
        if 'gemini_api_key' in config_data: config.gemini_api_key = self.encrypt(config_data['gemini_api_key'])
        if 'grok_api_key' in config_data: config.grok_api_key = self.encrypt(config_data['grok_api_key'])
        self.session.commit()
        
    def get_config(self, user_id):
        config = self.session.query(UserConfig).filter_by(user_id=user_id).first()
        if not config:
            return {}
        return {
            'db_host': self.decrypt(config.db_host) if config.db_host else "",
            'db_port': self.decrypt(config.db_port) if config.db_port else "",
            'db_name': self.decrypt(config.db_name) if config.db_name else "",
            'db_user': self.decrypt(config.db_user) if config.db_user else "",
            'db_password': self.decrypt(config.db_password) if config.db_password else "",
            'openai_api_key': self.decrypt(config.openai_api_key) if config.openai_api_key else "",
            'anthropic_api_key': self.decrypt(config.anthropic_api_key) if config.anthropic_api_key else "",
            'gemini_api_key': self.decrypt(config.gemini_api_key) if config.gemini_api_key else "",
            'grok_api_key': self.decrypt(config.grok_api_key) if config.grok_api_key else "",
        }

    # ========================
    # Dashboard Management
    # ========================
    
    def get_dashboard(self, user_id):
        dash = self.session.query(UserDashboard).filter_by(user_id=user_id).first()
        if dash:
            try:
                return json.loads(dash.data)
            except:
                pass
        return []

    def save_dashboard(self, user_id, data):
        dash = self.session.query(UserDashboard).filter_by(user_id=user_id).first()
        if not dash:
            dash = UserDashboard(user_id=user_id)
            self.session.add(dash)
        dash.data = json.dumps(data)
        self.session.commit()
