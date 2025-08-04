import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.process import Process
from crewai_tools import SerperDevTool
import os
import logging
import time
import subprocess
import threading
import queue
import platform
import json
import tempfile
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import whisper
import numpy as np
import wave
import io
from cryptography.fernet import Fernet
import pickle
import base64
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import keyring
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SecureCredentialManager:
    """
    Manages secure storage and retrieval of login credentials using encryption
    and system keyring with fallback to encrypted .env files.
    """
    
    def __init__(self, app_name: str = "MeetingAgent"):
        self.app_name = app_name
        self.credentials_dir = Path.home() / ".meeting_agent"
        self.credentials_dir.mkdir(exist_ok=True)
        self.encryption_key_file = self.credentials_dir / "encryption.key"
        self.credentials_file = self.credentials_dir / "credentials.enc"
        
        # Initialize encryption
        self._ensure_encryption_key()
        self.cipher = Fernet(self._load_encryption_key())
        
        self.logger = logging.getLogger(__name__ + ".Credentials")
    
    def _ensure_encryption_key(self):
        """Ensure encryption key exists, create if needed."""
        if not self.encryption_key_file.exists():
            key = Fernet.generate_key()
            
            # Try to store in system keyring first
            try:
                keyring.set_password(self.app_name, "encryption_key", key.decode())
                self.logger.info("Encryption key stored in system keyring")
            except Exception as e:
                self.logger.warning(f"Failed to store key in keyring: {e}")
                # Fallback to file storage
                with open(self.encryption_key_file, 'wb') as f:
                    f.write(key)
                self.logger.info("Encryption key stored in file")
    
    def _load_encryption_key(self) -> bytes:
        """Load encryption key from keyring or file."""
        try:
            # Try keyring first
            key_str = keyring.get_password(self.app_name, "encryption_key")
            if key_str:
                return key_str.encode()
        except Exception as e:
            self.logger.warning(f"Failed to load key from keyring: {e}")
        
        # Fallback to file
        if self.encryption_key_file.exists():
            with open(self.encryption_key_file, 'rb') as f:
                return f.read()
        
        raise ValueError("No encryption key found")
    
    def store_credentials(self, platform: str, email: str, password: str, 
                         additional_data: Dict[str, str] = None):
        """Store credentials securely for a platform."""
        credentials = {
            'email': email,
            'password': password,
            'platform': platform,
            'stored_at': datetime.now().isoformat()
        }
        
        if additional_data:
            credentials.update(additional_data)
        
        # Load existing credentials
        all_credentials = self._load_all_credentials()
        all_credentials[platform] = credentials
        
        # Encrypt and store
        encrypted_data = self.cipher.encrypt(pickle.dumps(all_credentials))
        with open(self.credentials_file, 'wb') as f:
            f.write(encrypted_data)
        
        self.logger.info(f"Credentials stored for platform: {platform}")
    
    def get_credentials(self, platform: str) -> Optional[Dict[str, str]]:
        """Retrieve credentials for a platform."""
        try:
            all_credentials = self._load_all_credentials()
            return all_credentials.get(platform)
        except Exception as e:
            self.logger.error(f"Failed to load credentials for {platform}: {e}")
            return None
    
    def _load_all_credentials(self) -> Dict[str, Dict]:
        """Load all stored credentials."""
        if not self.credentials_file.exists():
            return {}
        
        try:
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return pickle.loads(decrypted_data)
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {e}")
            return {}
    
    def remove_credentials(self, platform: str) -> bool:
        """Remove credentials for a platform."""
        try:
            all_credentials = self._load_all_credentials()
            if platform in all_credentials:
                del all_credentials[platform]
                
                encrypted_data = self.cipher.encrypt(pickle.dumps(all_credentials))
                with open(self.credentials_file, 'wb') as f:
                    f.write(encrypted_data)
                
                self.logger.info(f"Credentials removed for platform: {platform}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to remove credentials for {platform}: {e}")
        
        return False
    
    def list_stored_platforms(self) -> List[str]:
        """List platforms with stored credentials."""
        try:
            all_credentials = self._load_all_credentials()
            return list(all_credentials.keys())
        except Exception as e:
            self.logger.error(f"Failed to list platforms: {e}")
            return []
    
    @classmethod
    def from_env_file(cls, env_file: str = ".env") -> 'SecureCredentialManager':
        """Create manager and load credentials from .env file."""
        manager = cls()
        
        try:
            # Load .env file
            env_path = Path(env_file)
            if env_path.exists():
                from dotenv import dotenv_values
                env_vars = dotenv_values(env_path)
                
                # Store Google credentials if available
                google_email = env_vars.get('GOOGLE_EMAIL')
                google_password = env_vars.get('GOOGLE_PASSWORD')
                if google_email and google_password:
                    manager.store_credentials('google', google_email, google_password)
                
                # Store Zoom credentials if available
                zoom_email = env_vars.get('ZOOM_EMAIL')
                zoom_password = env_vars.get('ZOOM_PASSWORD')
                if zoom_email and zoom_password:
                    manager.store_credentials('zoom', zoom_email, zoom_password)
                
                manager.logger.info("Credentials loaded from .env file")
        
        except Exception as e:
            manager.logger.error(f"Failed to load from .env: {e}")
        
        return manager


class SessionManager:
    """
    Manages browser session persistence including cookies and local storage.
    """
    
    def __init__(self, session_dir: str = None):
        self.session_dir = Path(session_dir) if session_dir else Path.home() / ".meeting_agent" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__ + ".Session")
    
    def save_session(self, driver: webdriver.Chrome, platform: str, user_id: str = "default"):
        """Save browser session (cookies and local storage) for a platform."""
        try:
            session_file = self.session_dir / f"{platform}_{user_id}_session.json"
            
            # Get cookies
            cookies = driver.get_cookies()
            
            # Get local storage (if available)
            local_storage = {}
            try:
                local_storage = driver.execute_script(
                    "return Object.fromEntries(Object.entries(localStorage));"
                )
            except Exception as e:
                self.logger.warning(f"Could not access localStorage: {e}")
            
            # Get session storage (if available)
            session_storage = {}
            try:
                session_storage = driver.execute_script(
                    "return Object.fromEntries(Object.entries(sessionStorage));"
                )
            except Exception as e:
                self.logger.warning(f"Could not access sessionStorage: {e}")
            
            # Save session data
            session_data = {
                'cookies': cookies,
                'local_storage': local_storage,
                'session_storage': session_storage,
                'url': driver.current_url,
                'timestamp': datetime.now().isoformat(),
                'user_agent': driver.execute_script("return navigator.userAgent;")
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.logger.info(f"Session saved for {platform} user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
            return False
    
    def load_session(self, driver: webdriver.Chrome, platform: str, user_id: str = "default") -> bool:
        """Load browser session for a platform."""
        try:
            session_file = self.session_dir / f"{platform}_{user_id}_session.json"
            
            if not session_file.exists():
                self.logger.info(f"No saved session for {platform} user {user_id}")
                return False
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Check if session is too old (older than 7 days)
            session_time = datetime.fromisoformat(session_data['timestamp'])
            if (datetime.now() - session_time).days > 7:
                self.logger.info(f"Session for {platform} is too old, skipping")
                return False
            
            # Navigate to a page first (required for setting cookies)
            if platform == 'google':
                driver.get("https://accounts.google.com")
            elif platform == 'zoom':
                driver.get("https://zoom.us")
            else:
                driver.get(session_data.get('url', 'https://google.com'))
            
            # Wait for page load
            time.sleep(2)
            
            # Restore cookies
            for cookie in session_data.get('cookies', []):
                try:
                    driver.add_cookie(cookie)
                except Exception as e:
                    self.logger.warning(f"Failed to add cookie: {e}")
            
            # Restore local storage
            if session_data.get('local_storage'):
                try:
                    for key, value in session_data['local_storage'].items():
                        driver.execute_script(f"localStorage.setItem('{key}', '{value}');")
                except Exception as e:
                    self.logger.warning(f"Failed to restore localStorage: {e}")
            
            # Restore session storage
            if session_data.get('session_storage'):
                try:
                    for key, value in session_data['session_storage'].items():
                        driver.execute_script(f"sessionStorage.setItem('{key}', '{value}');")
                except Exception as e:
                    self.logger.warning(f"Failed to restore sessionStorage: {e}")
            
            self.logger.info(f"Session loaded for {platform} user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")
            return False
    
    def clear_session(self, platform: str, user_id: str = "default"):
        """Clear saved session for a platform."""
        try:
            session_file = self.session_dir / f"{platform}_{user_id}_session.json"
            if session_file.exists():
                session_file.unlink()
                self.logger.info(f"Session cleared for {platform} user {user_id}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to clear session: {e}")
        
        return False
    
    def list_saved_sessions(self) -> List[Dict[str, str]]:
        """List all saved sessions."""
        sessions = []
        try:
            for session_file in self.session_dir.glob("*_session.json"):
                parts = session_file.stem.replace('_session', '').split('_')
                if len(parts) >= 2:
                    platform = parts[0]
                    user_id = '_'.join(parts[1:])
                    
                    # Get timestamp
                    try:
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                        timestamp = session_data.get('timestamp', 'Unknown')
                    except:
                        timestamp = 'Unknown'
                    
                    sessions.append({
                        'platform': platform,
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'file': str(session_file)
                    })
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
        
        return sessions


class LoginHandler:
    """
    Handles login flows for different meeting platforms (Google, Zoom).
    Supports both OAuth and traditional form-based authentication.
    """
    
    def __init__(self, credentials_manager: SecureCredentialManager, 
                 session_manager: SessionManager,
                 logger: Optional[logging.Logger] = None):
        self.credentials_manager = credentials_manager
        self.session_manager = session_manager
        self.logger = logger or logging.getLogger(__name__ + ".Login")
        
        # Platform-specific login configurations
        self.login_configs = {
            'google': {
                'login_url': 'https://accounts.google.com/signin',
                'email_selector': 'input[type="email"]',
                'email_next_selector': '#identifierNext',
                'password_selector': 'input[type="password"]',
                'password_next_selector': '#passwordNext',
                'success_indicators': [
                    '[data-userprofile-email]',
                    '.gb_A',  # Profile image
                    '[aria-label*="Google Account"]'
                ],
                'oauth_domains': ['accounts.google.com', 'myaccount.google.com']
            },
            'zoom': {
                'login_url': 'https://zoom.us/signin',
                'email_selector': '#email',
                'password_selector': '#password',
                'submit_selector': '#login',
                'success_indicators': [
                    '.zm-avatar',
                    '.profile-pic',
                    '[data-user-id]'
                ],
                'oauth_domains': ['zoom.us']
            }
        }
    
    def attempt_login(self, driver: webdriver.Chrome, platform: str, 
                     force_fresh_login: bool = False) -> bool:
        """
        Attempt to log into a platform, trying session restore first, then credentials.
        """
        self.logger.info(f"Attempting login for platform: {platform}")
        
        # Step 1: Try to restore existing session (unless forced fresh)
        if not force_fresh_login:
            if self._try_session_restore(driver, platform):
                return True
        
        # Step 2: Try credential-based login
        return self._try_credential_login(driver, platform)
    
    def _try_session_restore(self, driver: webdriver.Chrome, platform: str) -> bool:
        """Try to restore an existing session."""
        try:
            self.logger.info(f"Attempting session restore for {platform}")
            
            # Load saved session
            if not self.session_manager.load_session(driver, platform):
                return False
            
            # Verify login status
            return self._verify_login_status(driver, platform)
            
        except Exception as e:
            self.logger.error(f"Session restore failed: {e}")
            return False
    
    def _try_credential_login(self, driver: webdriver.Chrome, platform: str) -> bool:
        """Try to login using stored credentials."""
        try:
            credentials = self.credentials_manager.get_credentials(platform)
            if not credentials:
                self.logger.warning(f"No credentials found for {platform}")
                return False
            
            return self._perform_login_flow(driver, platform, credentials)
            
        except Exception as e:
            self.logger.error(f"Credential login failed: {e}")
            return False
    
    def _perform_login_flow(self, driver: webdriver.Chrome, platform: str, 
                           credentials: Dict[str, str]) -> bool:
        """Perform the actual login flow for a platform."""
        config = self.login_configs.get(platform)
        if not config:
            self.logger.error(f"No login config for platform: {platform}")
            return False
        
        try:
            # Navigate to login page
            self.logger.info(f"Navigating to {platform} login page")
            driver.get(config['login_url'])
            time.sleep(3)
            
            if platform == 'google':
                return self._handle_google_login(driver, credentials, config)
            elif platform == 'zoom':
                return self._handle_zoom_login(driver, credentials, config)
            else:
                self.logger.error(f"Unsupported platform: {platform}")
                return False
                
        except Exception as e:
            self.logger.error(f"Login flow failed for {platform}: {e}")
            return False
    
    def _handle_google_login(self, driver: webdriver.Chrome, credentials: Dict[str, str], 
                            config: Dict[str, str]) -> bool:
        """Handle Google-specific login flow."""
        try:
            # Wait for email input
            email_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, config['email_selector']))
            )
            
            # Enter email
            email_input.clear()
            email_input.send_keys(credentials['email'])
            time.sleep(1)
            
            # Click Next
            next_button = driver.find_element(By.CSS_SELECTOR, config['email_next_selector'])
            next_button.click()
            time.sleep(3)
            
            # Wait for password input
            password_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, config['password_selector']))
            )
            
            # Enter password
            password_input.clear()
            password_input.send_keys(credentials['password'])
            time.sleep(1)
            
            # Click Next
            next_button = driver.find_element(By.CSS_SELECTOR, config['password_next_selector'])
            next_button.click()
            time.sleep(5)
            
            # Verify login success
            success = self._verify_login_status(driver, 'google')
            if success:
                # Save session for future use
                self.session_manager.save_session(driver, 'google', credentials['email'])
                self.logger.info("Google login successful, session saved")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Google login failed: {e}")
            return False
    
    def _handle_zoom_login(self, driver: webdriver.Chrome, credentials: Dict[str, str], 
                          config: Dict[str, str]) -> bool:
        """Handle Zoom-specific login flow."""
        try:
            # Wait for form elements
            email_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, config['email_selector']))
            )
            password_input = driver.find_element(By.CSS_SELECTOR, config['password_selector'])
            submit_button = driver.find_element(By.CSS_SELECTOR, config['submit_selector'])
            
            # Enter credentials
            email_input.clear()
            email_input.send_keys(credentials['email'])
            time.sleep(1)
            
            password_input.clear()
            password_input.send_keys(credentials['password'])
            time.sleep(1)
            
            # Submit form
            submit_button.click()
            time.sleep(5)
            
            # Verify login success
            success = self._verify_login_status(driver, 'zoom')
            if success:
                # Save session for future use
                self.session_manager.save_session(driver, 'zoom', credentials['email'])
                self.logger.info("Zoom login successful, session saved")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Zoom login failed: {e}")
            return False
    
    def _verify_login_status(self, driver: webdriver.Chrome, platform: str) -> bool:
        """Verify if login was successful by checking for platform-specific indicators."""
        config = self.login_configs.get(platform)
        if not config:
            return False
        
        try:
            # Check for success indicators
            for indicator in config['success_indicators']:
                try:
                    element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, indicator))
                    )
                    if element:
                        self.logger.info(f"Login verification successful for {platform}")
                        return True
                except TimeoutException:
                    continue
            
            # Additional verification: check URL domain
            current_url = driver.current_url
            for domain in config.get('oauth_domains', []):
                if domain in current_url and 'signin' not in current_url and 'login' not in current_url:
                    self.logger.info(f"Login verified by URL for {platform}")
                    return True
            
            self.logger.warning(f"Login verification failed for {platform}")
            return False
            
        except Exception as e:
            self.logger.error(f"Login verification error for {platform}: {e}")
            return False
    
    def logout(self, driver: webdriver.Chrome, platform: str) -> bool:
        """Logout from a platform and clear session."""
        try:
            if platform == 'google':
                driver.get('https://accounts.google.com/logout')
            elif platform == 'zoom':
                driver.get('https://zoom.us/signout')
            
            time.sleep(3)
            
            # Clear saved session
            self.session_manager.clear_session(platform)
            
            self.logger.info(f"Logged out from {platform}")
            return True
            
        except Exception as e:
            self.logger.error(f"Logout failed for {platform}: {e}")
            return False


class MeetingOrchestrator:
    """
    Orchestrates the entire meeting lifecycle including joining, audio capture, 
    transcription, and monitoring with retry logic and lifecycle management.
    """
    
    def __init__(self, 
                 joiner_config: Dict[str, Any],
                 max_join_attempts: int = 3,
                 join_confirmation_timeout: int = 60,
                 meeting_health_check_interval: int = 30,
                 audio_start_delay: int = 5,
                 enable_login: bool = True,
                 logger: Optional[logging.Logger] = None):
        
        self.joiner_config = joiner_config
        self.max_join_attempts = max_join_attempts
        self.join_confirmation_timeout = join_confirmation_timeout
        self.meeting_health_check_interval = meeting_health_check_interval
        self.audio_start_delay = audio_start_delay
        self.enable_login = enable_login
        
        self.logger = logger or logging.getLogger(__name__ + ".Orchestrator")
        
        # Authentication components
        self.credentials_manager = SecureCredentialManager()
        self.session_manager = SessionManager()
        self.login_handler = LoginHandler(self.credentials_manager, self.session_manager, self.logger)
        
        # State management
        self.joiner: Optional[MeetingAutoJoiner] = None
        self.meeting_url: Optional[str] = None
        self.current_state = "idle"  # idle, authenticating, joining, joined, recording, transcribing, ended, error
        self.state_history: List[Dict] = []
        self.join_attempts = 0
        self.last_health_check = 0
        self.login_platform: Optional[str] = None
        
        # Threading for lifecycle management
        self.lifecycle_thread: Optional[threading.Thread] = None
        self.stop_lifecycle_monitoring = False
        
        # Callbacks for state changes
        self.state_callbacks: Dict[str, List] = {}
    
    def add_state_callback(self, state: str, callback):
        """Add a callback function for when the orchestrator enters a specific state."""
        if state not in self.state_callbacks:
            self.state_callbacks[state] = []
        self.state_callbacks[state].append(callback)
    
    def _set_state(self, new_state: str, details: str = ""):
        """Set the current state and notify callbacks."""
        old_state = self.current_state
        self.current_state = new_state
        
        # Add to state history
        state_entry = {
            "timestamp": datetime.now(),
            "from_state": old_state,
            "to_state": new_state,
            "details": details
        }
        self.state_history.append(state_entry)
        
        self.logger.info(f"State transition: {old_state} → {new_state} ({details})")
        
        # Call state callbacks
        if new_state in self.state_callbacks:
            for callback in self.state_callbacks[new_state]:
                try:
                    callback(old_state, new_state, details)
                except Exception as e:
                    self.logger.error(f"Error in state callback: {e}")
    
    def start_meeting_session(self, meeting_url: str) -> bool:
        """
        Start a complete meeting session with orchestrated lifecycle management.
        """
        self.meeting_url = meeting_url
        self.join_attempts = 0
        self._set_state("joining", f"Starting session for {meeting_url}")
        
        # Start the orchestration lifecycle in a separate thread
        self.stop_lifecycle_monitoring = False
        self.lifecycle_thread = threading.Thread(
            target=self._orchestration_lifecycle,
            daemon=True
        )
        self.lifecycle_thread.start()
        
        return True
    
    def _orchestration_lifecycle(self):
        """Main orchestration lifecycle that manages the entire meeting session."""
        try:
            # Phase 0: Authentication (if enabled)
            if self.enable_login:
                if not self._handle_authentication():
                    self._set_state("error", "Authentication failed")
                    return
            
            # Phase 1: Join the meeting with retry logic
            if not self._join_with_retry():
                self._set_state("error", "Failed to join meeting after all attempts")
                return
            
            # Phase 2: Wait for join confirmation
            if not self._wait_for_join_confirmation():
                self._set_state("error", "Meeting join confirmation timeout")
                return
            
            # Phase 3: Start audio capture
            if not self._start_audio_capture_orchestrated():
                self._set_state("joined", "Meeting joined but audio capture failed")
            else:
                self._set_state("recording", "Audio capture active")
            
            # Phase 4: Start transcription if enabled
            if (self.joiner.enable_audio_capture and 
                self.joiner.real_time_transcription and 
                self.joiner.audio_manager):
                
                if self._start_transcription_orchestrated():
                    self._set_state("transcribing", "Real-time transcription active")
            
            # Phase 5: Monitor meeting health and lifecycle
            self._monitor_meeting_lifecycle()
            
        except Exception as e:
            self.logger.error(f"Error in orchestration lifecycle: {e}")
            self._set_state("error", f"Lifecycle error: {str(e)}")
    
    def _handle_authentication(self) -> bool:
        """Handle authentication for the meeting platform."""
        try:
            self._set_state("authenticating", "Determining platform and authenticating")
            
            # Determine platform from URL
            self.login_platform = self._detect_platform_from_url(self.meeting_url)
            
            if self.login_platform not in ['google', 'zoom']:
                self.logger.info(f"Platform {self.login_platform} doesn't require authentication")
                return True
            
            # Create temporary driver for authentication
            temp_joiner = MeetingAutoJoiner(**self.joiner_config)
            temp_driver = temp_joiner._setup_browser()
            
            try:
                # Attempt login
                login_success = self.login_handler.attempt_login(temp_driver, self.login_platform)
                
                if login_success:
                    self.logger.info(f"Authentication successful for {self.login_platform}")
                    # Save the authenticated session
                    credentials = self.credentials_manager.get_credentials(self.login_platform)
                    if credentials:
                        user_id = credentials.get('email', 'default')
                        self.session_manager.save_session(temp_driver, self.login_platform, user_id)
                    return True
                else:
                    self.logger.warning(f"Authentication failed for {self.login_platform}")
                    return False
                    
            finally:
                # Clean up temporary driver
                try:
                    temp_driver.quit()
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    def _detect_platform_from_url(self, url: str) -> str:
        """Detect platform from meeting URL."""
        url_lower = url.lower()
        if 'meet.google.com' in url_lower:
            return 'google'
        elif 'zoom.us' in url_lower or 'zoom.com' in url_lower:
            return 'zoom'
        else:
            return 'unknown'
    
    def _join_with_retry(self) -> bool:
        """Join meeting with retry logic and exponential backoff."""
        for attempt in range(1, self.max_join_attempts + 1):
            self.join_attempts = attempt
            self._set_state("joining", f"Join attempt {attempt}/{self.max_join_attempts}")
            
            try:
                # Create fresh joiner instance for each attempt
                if self.joiner:
                    try:
                        self.joiner.leave_meeting()
                    except:
                        pass
                
                # Pass authentication managers to joiner
                joiner_config_with_auth = self.joiner_config.copy()
                joiner_config_with_auth['session_manager'] = self.session_manager
                joiner_config_with_auth['login_platform'] = self.login_platform
                
                self.joiner = MeetingAutoJoiner(**joiner_config_with_auth)
                
                # Attempt to join
                success = self.joiner.join_meeting(self.meeting_url)
                
                if success:
                    self.logger.info(f"Successfully joined meeting on attempt {attempt}")
                    return True
                else:
                    self.logger.warning(f"Join attempt {attempt} failed")
                    
            except Exception as e:
                self.logger.error(f"Join attempt {attempt} exception: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_join_attempts:
                wait_time = min(5 * (2 ** (attempt - 1)), 30)  # Cap at 30 seconds
                self.logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        return False
    
    def _wait_for_join_confirmation(self) -> bool:
        """Wait for confirmation that the meeting join was successful."""
        self.logger.info("Waiting for join confirmation...")
        start_time = time.time()
        
        while time.time() - start_time < self.join_confirmation_timeout:
            try:
                if self.joiner and self.joiner.driver:
                    # Check for meeting indicators
                    meeting_confirmed = self._check_meeting_indicators()
                    
                    if meeting_confirmed:
                        self.logger.info("Meeting join confirmed")
                        return True
                    
                    # Also check if we're still in a valid meeting state
                    if not self.joiner.is_in_meeting():
                        # Wait a bit more as some platforms take time to load
                        if time.time() - start_time > 20:  # Give it at least 20 seconds
                            self.logger.warning("Join confirmation failed - not in meeting")
                            return False
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error during join confirmation: {e}")
                time.sleep(2)
        
        self.logger.warning("Join confirmation timeout")
        return False
    
    def _check_meeting_indicators(self) -> bool:
        """Check for specific indicators that confirm we're in a meeting."""
        if not self.joiner or not self.joiner.driver:
            return False
        
        try:
            # Google Meet indicators
            googlemeet_indicators = [
                '[data-call-status="in-call"]',
                '.call-view',
                '[data-meeting-title]',
                '.google-material-icons[data-is-muted]',  # Mute button
                '[data-tooltip="Turn camera off"]',        # Camera button
            ]
            
            # Zoom indicators  
            zoom_indicators = [
                '.meeting-client-inner',
                '.meeting-client-content',
                '.footer-button__wrapper',
                '.zm-video-container',
                '#wc-container-right',
            ]
            
            # Platform-agnostic indicators
            general_indicators = [
                '[aria-label*="mute"]',
                '[aria-label*="camera"]',
                '[aria-label*="video"]',
                '[role="button"][aria-label*="Leave"]',
            ]
            
            all_indicators = googlemeet_indicators + zoom_indicators + general_indicators
            
            for indicator in all_indicators:
                try:
                    elements = self.joiner.driver.find_elements(By.CSS_SELECTOR, indicator)
                    if elements:
                        self.logger.debug(f"Found meeting indicator: {indicator}")
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking meeting indicators: {e}")
            return False
    
    def _start_audio_capture_orchestrated(self) -> bool:
        """Start audio capture with proper orchestration."""
        if not self.joiner or not self.joiner.enable_audio_capture:
            return False
        
        try:
            self.logger.info(f"Waiting {self.audio_start_delay}s before starting audio capture...")
            time.sleep(self.audio_start_delay)
            
            if self.joiner.audio_manager:
                success = self.joiner.audio_manager.start_recording(self.joiner.audio_capture_mode)
                if success:
                    self.logger.info("Audio capture started successfully")
                    return True
                else:
                    self.logger.error("Failed to start audio capture")
                    return False
            
        except Exception as e:
            self.logger.error(f"Error starting audio capture: {e}")
            return False
        
        return False
    
    def _start_transcription_orchestrated(self) -> bool:
        """Start transcription with proper orchestration."""
        try:
            if (self.joiner and 
                self.joiner.audio_manager and 
                self.joiner.audio_manager.is_recording):
                
                # Wait a bit for audio to stabilize
                time.sleep(2)
                
                self.joiner.audio_manager.start_transcription()
                self.logger.info("Transcription started successfully")
                return True
            else:
                self.logger.warning("Cannot start transcription: audio not recording")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting transcription: {e}")
            return False
    
    def _monitor_meeting_lifecycle(self):
        """Monitor the meeting lifecycle and handle state changes."""
        self.logger.info("Starting meeting lifecycle monitoring")
        
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while not self.stop_lifecycle_monitoring:
            try:
                current_time = time.time()
                
                # Perform health check
                if current_time - self.last_health_check > self.meeting_health_check_interval:
                    meeting_active = self._perform_health_check()
                    
                    if meeting_active:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        self.logger.warning(f"Health check failed ({consecutive_failures}/{max_consecutive_failures})")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            self._set_state("ended", "Meeting ended (health check failures)")
                            break
                    
                    self.last_health_check = current_time
                
                # Check for browser/driver issues
                if self.joiner and not self.joiner.driver:
                    self._set_state("error", "Browser driver lost")
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in lifecycle monitoring: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    self._set_state("error", f"Monitoring failed: {str(e)}")
                    break
                time.sleep(5)
        
        self.logger.info("Meeting lifecycle monitoring ended")
    
    def _perform_health_check(self) -> bool:
        """Perform a comprehensive health check of the meeting session."""
        if not self.joiner:
            return False
        
        try:
            # Check if still in meeting
            in_meeting = self.joiner.is_in_meeting()
            
            # Check if browser is responsive
            browser_responsive = self._check_browser_responsive()
            
            # Check audio capture if enabled
            audio_healthy = True
            if self.joiner.audio_manager and self.joiner.enable_audio_capture:
                audio_healthy = self.joiner.audio_manager.is_recording
            
            health_status = in_meeting and browser_responsive and audio_healthy
            
            if not health_status:
                details = []
                if not in_meeting:
                    details.append("not in meeting")
                if not browser_responsive:
                    details.append("browser unresponsive")
                if not audio_healthy:
                    details.append("audio capture failed")
                
                self.logger.warning(f"Health check failed: {', '.join(details)}")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return False
    
    def _check_browser_responsive(self) -> bool:
        """Check if the browser is still responsive."""
        if not self.joiner or not self.joiner.driver:
            return False
        
        try:
            # Simple check - get current URL
            current_url = self.joiner.driver.current_url
            return bool(current_url)
        except:
            return False
    
    def force_reconnect(self) -> bool:
        """Force a reconnection to the meeting."""
        if not self.meeting_url:
            return False
        
        self.logger.info("Forcing reconnection...")
        self._set_state("joining", "Force reconnecting")
        
        # Stop current session
        self.stop_session()
        
        # Wait a bit
        time.sleep(2)
        
        # Restart session
        return self.start_meeting_session(self.meeting_url)
    
    def stop_session(self):
        """Stop the entire meeting session and cleanup."""
        self.logger.info("Stopping meeting session")
        
        # Stop lifecycle monitoring
        self.stop_lifecycle_monitoring = True
        
        # Stop audio and transcription
        if self.joiner:
            try:
                self.joiner.leave_meeting()
            except Exception as e:
                self.logger.error(f"Error during session cleanup: {e}")
        
        # Wait for lifecycle thread to finish
        if self.lifecycle_thread and self.lifecycle_thread.is_alive():
            self.lifecycle_thread.join(timeout=5)
        
        self._set_state("idle", "Session stopped")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        stats = {
            "current_state": self.current_state,
            "join_attempts": self.join_attempts,
            "meeting_url": self.meeting_url,
            "session_duration": 0,
            "audio_active": False,
            "transcription_active": False,
            "transcript_segments": 0,
            "state_history": self.state_history[-10:]  # Last 10 state changes
        }
        
        # Calculate session duration
        if self.state_history:
            first_join = next((s for s in self.state_history if s["to_state"] == "joining"), None)
            if first_join:
                stats["session_duration"] = (datetime.now() - first_join["timestamp"]).total_seconds()
        
        # Audio and transcription status
        if self.joiner and self.joiner.audio_manager:
            stats["audio_active"] = self.joiner.audio_manager.is_recording
            stats["transcription_active"] = self.joiner.audio_manager.is_transcribing
            stats["transcript_segments"] = len(self.joiner.audio_manager.transcripts)
        
        return stats
    
    def get_current_state(self) -> str:
        """Get the current orchestration state."""
        return self.current_state
    
    def is_session_active(self) -> bool:
        """Check if the session is currently active."""
        return self.current_state in ["joining", "joined", "recording", "transcribing"]


class AudioCaptureManager:
    """
    Cross-platform audio capture manager using FFMPEG for system audio recording.
    Supports real-time transcription using Whisper.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_duration: int = 5,
                 transcription_backend: str = "whisper-local",
                 whisper_model: str = "base"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.transcription_backend = transcription_backend
        self.whisper_model = whisper_model
        
        # Threading and process management
        self.audio_process: Optional[subprocess.Popen] = None
        self.transcription_thread: Optional[threading.Thread] = None
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.is_recording = False
        self.is_transcribing = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__ + ".AudioCapture")
        
        # Load Whisper model if using local transcription
        self.whisper_model_instance = None
        if transcription_backend == "whisper-local":
            try:
                self.whisper_model_instance = whisper.load_model(whisper_model)
                self.logger.info(f"Loaded Whisper model: {whisper_model}")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
        
        # Store captured audio and transcripts
        self.audio_chunks: List[bytes] = []
        self.transcripts: List[Dict] = []
    
    def _get_system_audio_command(self) -> List[str]:
        """
        Get the appropriate FFMPEG command for system audio capture based on the platform.
        """
        system = platform.system().lower()
        
        if system == "windows":
            # Windows WASAPI - capture system audio
            return [
                "ffmpeg",
                "-f", "dshow",
                "-i", "audio=Stereo Mix",  # Default system audio device
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-f", "wav",
                "-"
            ]
        
        elif system == "darwin":  # macOS
            # macOS Core Audio - requires BlackHole or similar virtual audio device
            return [
                "ffmpeg",
                "-f", "avfoundation",
                "-i", ":0",  # Default audio input
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-f", "wav",
                "-"
            ]
        
        elif system == "linux":
            # Linux PulseAudio
            return [
                "ffmpeg",
                "-f", "pulse",
                "-i", "default",
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-f", "wav",
                "-"
            ]
        
        else:
            raise ValueError(f"Unsupported platform: {system}")
    
    def _get_browser_audio_command(self, browser_process_name: str = "chrome") -> List[str]:
        """
        Get FFMPEG command to capture audio from a specific browser process.
        This is more targeted than system-wide capture.
        """
        system = platform.system().lower()
        
        if system == "windows":
            # Use application-specific capture on Windows
            return [
                "ffmpeg",
                "-f", "dshow",
                "-i", f"audio={browser_process_name}",
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-f", "wav",
                "-"
            ]
        
        elif system == "darwin":
            # macOS - capture from specific application
            return [
                "ffmpeg",
                "-f", "avfoundation",
                "-i", ":0",  # Will need to be modified for app-specific capture
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-f", "wav",
                "-"
            ]
        
        elif system == "linux":
            # Linux - use pactl to find and capture from specific application
            return [
                "ffmpeg",
                "-f", "pulse",
                "-i", "default",  # Could be made more specific
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-f", "wav",
                "-"
            ]
        
        else:
            return self._get_system_audio_command()
    
    def _check_ffmpeg_available(self) -> bool:
        """Check if FFMPEG is available on the system."""
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _setup_virtual_audio_instructions(self) -> Dict[str, str]:
        """
        Provide platform-specific instructions for setting up virtual audio devices.
        """
        system = platform.system().lower()
        
        instructions = {
            "windows": """
            For Windows audio capture, you'll need:
            1. Enable 'Stereo Mix' in Sound Settings:
               - Right-click sound icon → Sounds → Recording tab
               - Right-click empty space → Show Disabled Devices
               - Enable 'Stereo Mix' device
            
            Alternative: Install VB-Audio Cable:
            - Download from: https://vb-audio.com/Cable/
            - Set as default playback device for system audio
            """,
            
            "darwin": """
            For macOS audio capture, install BlackHole:
            1. Download from: https://github.com/ExistentialAudio/BlackHole
            2. Install the 2ch version
            3. Create Multi-Output Device in Audio MIDI Setup:
               - Open Audio MIDI Setup app
               - Create Multi-Output Device
               - Select both built-in output and BlackHole 2ch
            4. Set Multi-Output Device as system output
            """,
            
            "linux": """
            For Linux audio capture:
            1. PulseAudio should work out of the box
            2. If needed, install pulseaudio-utils:
               sudo apt-get install pulseaudio-utils
            3. List available sources:
               pactl list sources short
            4. Use monitor source for system audio capture
            """
        }
        
        return instructions.get(system, "Platform not supported")
    
    def start_recording(self, capture_mode: str = "system") -> bool:
        """
        Start audio recording process.
        
        Args:
            capture_mode: "system" for system-wide audio, "browser" for browser-specific
        """
        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return False
        
        if not self._check_ffmpeg_available():
            self.logger.error("FFMPEG not available. Please install FFMPEG.")
            return False
        
        try:
            # Get appropriate command based on capture mode
            if capture_mode == "browser":
                cmd = self._get_browser_audio_command()
            else:
                cmd = self._get_system_audio_command()
            
            self.logger.info(f"Starting audio capture with command: {' '.join(cmd)}")
            
            # Start FFMPEG process
            self.audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.is_recording = True
            
            # Start audio reading thread
            self.transcription_thread = threading.Thread(
                target=self._audio_capture_loop,
                daemon=True
            )
            self.transcription_thread.start()
            
            self.logger.info("Audio capture started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def _audio_capture_loop(self):
        """Main loop for capturing and processing audio chunks."""
        chunk_size = self.sample_rate * self.channels * self.chunk_duration * 2  # 2 bytes per sample
        
        while self.is_recording and self.audio_process:
            try:
                # Read audio chunk from FFMPEG
                audio_data = self.audio_process.stdout.read(chunk_size)
                
                if not audio_data:
                    break
                
                # Store audio chunk
                self.audio_chunks.append(audio_data)
                
                # Add to processing queue
                self.audio_queue.put(audio_data)
                
                # Process transcription if enabled
                if self.is_transcribing:
                    self._process_audio_chunk(audio_data)
                    
            except Exception as e:
                self.logger.error(f"Error in audio capture loop: {e}")
                break
    
    def _process_audio_chunk(self, audio_data: bytes):
        """Process audio chunk for transcription."""
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Skip silent chunks (basic voice activity detection)
            if np.abs(audio_np).mean() < 0.001:
                return
            
            # Transcribe using Whisper
            if self.whisper_model_instance:
                result = self.whisper_model_instance.transcribe(
                    audio_np, 
                    language="en",
                    task="transcribe"
                )
                
                text = result.get("text", "").strip()
                if text:
                    transcript_entry = {
                        "timestamp": datetime.now(),
                        "text": text,
                        "confidence": result.get("confidence", 0.0)
                    }
                    
                    self.transcripts.append(transcript_entry)
                    self.transcript_queue.put(transcript_entry)
                    
                    self.logger.debug(f"Transcribed: {text}")
                    
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
    
    def start_transcription(self):
        """Enable real-time transcription."""
        self.is_transcribing = True
        self.logger.info("Real-time transcription enabled")
    
    def stop_transcription(self):
        """Disable real-time transcription."""
        self.is_transcribing = False
        self.logger.info("Real-time transcription disabled")
    
    def stop_recording(self):
        """Stop audio recording and cleanup."""
        self.is_recording = False
        self.is_transcribing = False
        
        if self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.audio_process.kill()
            finally:
                self.audio_process = None
        
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2)
        
        self.logger.info("Audio recording stopped")
    
    def get_latest_transcripts(self, count: int = 10) -> List[Dict]:
        """Get the latest transcription results."""
        return self.transcripts[-count:] if self.transcripts else []
    
    def get_full_transcript(self) -> str:
        """Get the complete transcript as a single string."""
        return "\n".join([t["text"] for t in self.transcripts])
    
    def save_audio_to_file(self, filepath: str) -> bool:
        """Save captured audio to a WAV file."""
        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                
                for chunk in self.audio_chunks:
                    wav_file.writeframes(chunk)
            
            self.logger.info(f"Audio saved to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            return False
    
    def export_transcript(self, filepath: str, format: str = "txt") -> bool:
        """Export transcript to file in various formats."""
        try:
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(self.transcripts, f, indent=2, default=str)
            
            elif format.lower() == "txt":
                with open(filepath, 'w') as f:
                    f.write(self.get_full_transcript())
            
            elif format.lower() == "srt":
                # Create SRT subtitle format
                with open(filepath, 'w') as f:
                    for i, transcript in enumerate(self.transcripts, 1):
                        timestamp = transcript["timestamp"]
                        start_time = timestamp.strftime("%H:%M:%S,000")
                        # Estimate end time (duration of chunk)
                        end_time = (timestamp.replace(second=timestamp.second + self.chunk_duration)).strftime("%H:%M:%S,000")
                        
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{transcript['text']}\n\n")
            
            self.logger.info(f"Transcript exported to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export transcript: {e}")
            return False


class MeetingAutoJoiner:
    """
    A reusable class for automatically joining Google Meet or Zoom meetings.
    """
    
    # Platform-specific configurations
    PLATFORM_CONFIGS = {
        'googlemeet': {
            'name': 'Google Meet',
            'join_selectors': [
                '[data-promo-anchor-id="start_meeting"]',
                '[jsname="Qx7uuf"]',
                'div[role="button"][aria-label*="Join"]',
                'div[role="button"]:has-text("Join now")',
                'div[data-call-action="join"]',
            ],
            'name_input_selectors': [
                'input[placeholder*="name"]',
                'input[aria-label*="name"]',
                '#join-dialog input[type="text"]',
            ],
            'camera_button_selectors': [
                '[data-is-muted="false"][aria-label*="camera"]',
                '[aria-label*="Turn off camera"]',
                'div[role="button"][aria-label*="camera"]',
            ],
            'mic_button_selectors': [
                '[data-is-muted="false"][aria-label*="microphone"]',
                '[aria-label*="Turn off microphone"]',
                'div[role="button"][aria-label*="microphone"]',
            ]
        },
        'zoom': {
            'name': 'Zoom',
            'join_selectors': [
                '#joinBtn',
                'a[onclick*="join"]',
                '.join-dialog .btn-primary',
                'button[class*="join"]',
                '.zm-btn--primary',
            ],
            'name_input_selectors': [
                '#inputname',
                'input[placeholder*="name"]',
                '#display-name-input',
            ]
        }
    }
    
    def __init__(self, 
                 headless: bool = True,
                 browser_profile_path: Optional[str] = None,
                 display_name: str = "Meeting Participant",
                 auto_mute_camera: bool = True,
                 auto_mute_microphone: bool = True,
                 wait_timeout: int = 30,
                 log_level: str = "INFO",
                 # Audio capture options
                 enable_audio_capture: bool = False,
                 audio_capture_mode: str = "system",
                 whisper_model: str = "base",
                 real_time_transcription: bool = False,
                 # Authentication options
                 session_manager: Optional[SessionManager] = None,
                 login_platform: Optional[str] = None):
        self.headless = headless
        self.browser_profile_path = browser_profile_path
        self.display_name = display_name
        self.auto_mute_camera = auto_mute_camera
        self.auto_mute_microphone = auto_mute_microphone
        self.wait_timeout = wait_timeout
        self.enable_audio_capture = enable_audio_capture
        self.audio_capture_mode = audio_capture_mode
        self.real_time_transcription = real_time_transcription
        
        # Authentication components
        self.session_manager = session_manager
        self.login_platform = login_platform
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)
        
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        
        # Audio capture manager
        self.audio_manager: Optional[AudioCaptureManager] = None
        if enable_audio_capture:
            try:
                self.audio_manager = AudioCaptureManager(
                    whisper_model=whisper_model,
                    transcription_backend="whisper-local"
                )
                self.logger.info("Audio capture manager initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize audio manager: {e}")
                self.audio_manager = None
    
    def _setup_browser(self) -> webdriver.Chrome:
        """Setup and configure the Chrome browser with session restoration."""
        options = Options()
        
        if self.headless:
            options.add_argument('--headless=new')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Media permissions
        prefs = {
            "profile.default_content_setting_values": {
                "media_stream_camera": 1,
                "media_stream_mic": 1,
                "notifications": 2
            }
        }
        options.add_experimental_option("prefs", prefs)
        
        if self.browser_profile_path:
            options.add_argument(f'--user-data-dir={self.browser_profile_path}')
        
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Restore session if available
            if self.session_manager and self.login_platform:
                try:
                    session_restored = self.session_manager.load_session(driver, self.login_platform)
                    if session_restored:
                        self.logger.info(f"Session restored for {self.login_platform}")
                    else:
                        self.logger.info(f"No session to restore for {self.login_platform}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore session: {e}")
            
            self.logger.info("Browser setup completed successfully")
            return driver
        except Exception as e:
            self.logger.error(f"Failed to setup browser: {e}")
            raise
    
    def _detect_platform(self, url: str) -> str:
        """Detect the meeting platform from the URL."""
        url_lower = url.lower()
        if 'meet.google.com' in url_lower or 'meet.google' in url_lower:
            return 'googlemeet'
        elif 'zoom.us' in url_lower or 'zoom.com' in url_lower:
            return 'zoom'
        else:
            self.logger.warning(f"Unknown platform for URL: {url}")
            return 'unknown'
    
    def _wait_and_find_element(self, selectors: list, timeout: Optional[int] = None):
        """Try to find an element using multiple selectors."""
        timeout = timeout or self.wait_timeout
        
        for selector in selectors:
            try:
                element = WebDriverWait(self.driver, timeout).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                self.logger.debug(f"Found element with selector: {selector}")
                return element
            except TimeoutException:
                continue
        return None
    
    def _handle_google_meet_join(self) -> bool:
        """Handle joining a Google Meet meeting."""
        self.logger.info("Handling Google Meet join process")
        config = self.PLATFORM_CONFIGS['googlemeet']
        
        try:
            time.sleep(3)
            
            # Enter name if required
            name_input = self._wait_and_find_element(config['name_input_selectors'], timeout=10)
            if name_input:
                self.logger.info("Entering display name")
                name_input.clear()
                name_input.send_keys(self.display_name)
                time.sleep(1)
            
            # Handle camera/microphone muting
            if self.auto_mute_camera:
                camera_button = self._wait_and_find_element(config['camera_button_selectors'], timeout=5)
                if camera_button:
                    self.logger.info("Muting camera")
                    camera_button.click()
                    time.sleep(1)
            
            if self.auto_mute_microphone:
                mic_button = self._wait_and_find_element(config['mic_button_selectors'], timeout=5)
                if mic_button:
                    self.logger.info("Muting microphone")
                    mic_button.click()
                    time.sleep(1)
            
            # Click join button
            join_button = self._wait_and_find_element(config['join_selectors'])
            if join_button:
                self.logger.info("Clicking join button")
                join_button.click()
                time.sleep(3)
                return True
            else:
                self.logger.error("Could not find join button")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in Google Meet join process: {e}")
            return False
    
    def _handle_zoom_join(self) -> bool:
        """Handle joining a Zoom meeting."""
        self.logger.info("Handling Zoom join process")
        config = self.PLATFORM_CONFIGS['zoom']
        
        try:
            time.sleep(3)
            
            # Check for "Join from Browser"
            join_browser_links = [
                'a[href*="join"]',
                '.join-from-browser',
                '#join-from-browser'
            ]
            
            browser_join = self._wait_and_find_element(join_browser_links, timeout=10)
            if browser_join:
                self.logger.info("Clicking 'Join from Browser'")
                browser_join.click()
                time.sleep(3)
            
            # Enter name
            name_input = self._wait_and_find_element(config['name_input_selectors'], timeout=10)
            if name_input:
                self.logger.info("Entering display name")
                name_input.clear()
                name_input.send_keys(self.display_name)
                time.sleep(1)
            
            # Click join button
            join_button = self._wait_and_find_element(config['join_selectors'])
            if join_button:
                self.logger.info("Clicking join button")
                join_button.click()
                time.sleep(3)
                return True
            else:
                self.logger.error("Could not find join button")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in Zoom join process: {e}")
            return False
    
    def join_meeting(self, meeting_url: str) -> bool:
        """Join a meeting using the provided URL. Audio capture is handled separately by orchestrator."""
        try:
            self.driver = self._setup_browser()
            self.wait = WebDriverWait(self.driver, self.wait_timeout)
            
            platform = self._detect_platform(meeting_url)
            self.logger.info(f"Detected platform: {platform}")
            
            self.logger.info(f"Navigating to meeting URL: {meeting_url}")
            self.driver.get(meeting_url)
            
            if platform == 'googlemeet':
                success = self._handle_google_meet_join()
            elif platform == 'zoom':
                success = self._handle_zoom_join()
            else:
                self.logger.error(f"Unsupported platform: {platform}")
                return False
            
            if success:
                self.logger.info("Successfully joined the meeting!")
                return True
            else:
                self.logger.error("Failed to join the meeting")
                return False
                
        except Exception as e:
            self.logger.error(f"Error joining meeting: {e}")
            return False
    
    def start_transcription(self):
        """Start real-time transcription if audio capture is active."""
        if self.audio_manager and self.audio_manager.is_recording:
            self.audio_manager.start_transcription()
            self.logger.info("Real-time transcription started")
            return True
        else:
            self.logger.warning("Cannot start transcription: audio capture not active")
            return False
    
    def stop_transcription(self):
        """Stop real-time transcription."""
        if self.audio_manager:
            self.audio_manager.stop_transcription()
            self.logger.info("Real-time transcription stopped")
            return True
        return False
    
    def get_live_transcript(self, count: int = 10) -> List[Dict]:
        """Get the latest transcription results."""
        if self.audio_manager:
            return self.audio_manager.get_latest_transcripts(count)
        return []
    
    def get_full_transcript(self) -> str:
        """Get the complete meeting transcript."""
        if self.audio_manager:
            return self.audio_manager.get_full_transcript()
        return ""
    
    def save_meeting_audio(self, filepath: str) -> bool:
        """Save the captured meeting audio to a file."""
        if self.audio_manager:
            return self.audio_manager.save_audio_to_file(filepath)
        return False
    
    def export_meeting_transcript(self, filepath: str, format: str = "txt") -> bool:
        """Export meeting transcript to file."""
        if self.audio_manager:
            return self.audio_manager.export_transcript(filepath, format)
        return False
    
    def get_audio_setup_instructions(self) -> str:
        """Get platform-specific audio setup instructions."""
        if self.audio_manager:
            return self.audio_manager._setup_virtual_audio_instructions()
        return "Audio capture not enabled"
    
    def leave_meeting(self):
        """Leave the meeting and close the browser."""
        # Stop audio capture first
        if self.audio_manager and self.audio_manager.is_recording:
            try:
                self.logger.info("Stopping audio capture")
                self.audio_manager.stop_recording()
            except Exception as e:
                self.logger.error(f"Error stopping audio capture: {e}")
        
        # Close browser
        if self.driver:
            try:
                self.logger.info("Leaving meeting and closing browser")
                self.driver.quit()
            except Exception as e:
                self.logger.error(f"Error closing browser: {e}")
    
    def is_in_meeting(self) -> bool:
        """Check if currently in a meeting."""
        if not self.driver:
            return False
        
        try:
            meeting_indicators = [
                '[data-call-status="in-call"]',
                '.meeting-client-inner',
                '.call-view',
                '.meeting-client-content',
            ]
            
            for indicator in meeting_indicators:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, indicator)
                    if element:
                        return True
                except NoSuchElementException:
                    continue
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking meeting status: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.leave_meeting()


# Streamlit app setup
st.set_page_config(page_title="AI Meeting Agent with Smart Orchestration 🎭", layout="wide")
st.title("AI Meeting Agent with Smart Orchestration 🎭")

# Sidebar for API keys
st.sidebar.header("API Keys")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
serper_api_key = st.sidebar.text_input("Serper API Key", type="password")

# Check if all API keys are set
if anthropic_api_key and serper_api_key:
    # # Set API keys as environment variables
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    claude = LLM(model="claude-3-5-sonnet-20240620", temperature= 0.7, api_key=anthropic_api_key)
    search_tool = SerperDevTool()

    # Input fields
    company_name = st.text_input("Enter the company name:")
    meeting_objective = st.text_input("Enter the meeting objective:")
    attendees = st.text_area("Enter the attendees and their roles (one per line):")
    meeting_duration = st.number_input("Enter the meeting duration (in minutes):", min_value=15, max_value=180, value=60, step=15)
    focus_areas = st.text_input("Enter any specific areas of focus or concerns:")

    # Define the agents
    context_analyzer = Agent(
        role='Meeting Context Specialist',
        goal='Analyze and summarize key background information for the meeting',
        backstory='You are an expert at quickly understanding complex business contexts and identifying critical information.',
        verbose=True,
        allow_delegation=False,
        llm=claude,
        tools=[search_tool]
    )

    industry_insights_generator = Agent(
        role='Industry Expert',
        goal='Provide in-depth industry analysis and identify key trends',
        backstory='You are a seasoned industry analyst with a knack for spotting emerging trends and opportunities.',
        verbose=True,
        allow_delegation=False,
        llm=claude,
        tools=[search_tool]
    )

    strategy_formulator = Agent(
        role='Meeting Strategist',
        goal='Develop a tailored meeting strategy and detailed agenda',
        backstory='You are a master meeting planner, known for creating highly effective strategies and agendas.',
        verbose=True,
        allow_delegation=False,
        llm=claude,
    )

    executive_briefing_creator = Agent(
        role='Communication Specialist',
        goal='Synthesize information into concise and impactful briefings',
        backstory='You are an expert communicator, skilled at distilling complex information into clear, actionable insights.',
        verbose=True,
        allow_delegation=False,
        llm=claude,
    )

    # Define the tasks
    context_analysis_task = Task(
        description=f"""
        Analyze the context for the meeting with {company_name}, considering:
        1. The meeting objective: {meeting_objective}
        2. The attendees: {attendees}
        3. The meeting duration: {meeting_duration} minutes
        4. Specific focus areas or concerns: {focus_areas}

        Research {company_name} thoroughly, including:
        1. Recent news and press releases
        2. Key products or services
        3. Major competitors

        Provide a comprehensive summary of your findings, highlighting the most relevant information for the meeting context.
        Format your output using markdown with appropriate headings and subheadings.
        """,
        agent=context_analyzer,
        expected_output="A detailed analysis of the meeting context and company background, including recent developments, financial performance, and relevance to the meeting objective, formatted in markdown with headings and subheadings."
    )

    industry_analysis_task = Task(
        description=f"""
        Based on the context analysis for {company_name} and the meeting objective: {meeting_objective}, provide an in-depth industry analysis:
        1. Identify key trends and developments in the industry
        2. Analyze the competitive landscape
        3. Highlight potential opportunities and threats
        4. Provide insights on market positioning

        Ensure the analysis is relevant to the meeting objective and attendees' roles.
        Format your output using markdown with appropriate headings and subheadings.
        """,
        agent=industry_insights_generator,
        expected_output="A comprehensive industry analysis report, including trends, competitive landscape, opportunities, threats, and relevant insights for the meeting objective, formatted in markdown with headings and subheadings."
    )

    strategy_development_task = Task(
        description=f"""
        Using the context analysis and industry insights, develop a tailored meeting strategy and detailed agenda for the {meeting_duration}-minute meeting with {company_name}. Include:
        1. A time-boxed agenda with clear objectives for each section
        2. Key talking points for each agenda item
        3. Suggested speakers or leaders for each section
        4. Potential discussion topics and questions to drive the conversation
        5. Strategies to address the specific focus areas and concerns: {focus_areas}

        Ensure the strategy and agenda align with the meeting objective: {meeting_objective}
        Format your output using markdown with appropriate headings and subheadings.
        """,
        agent=strategy_formulator,
        expected_output="A detailed meeting strategy and time-boxed agenda, including objectives, key talking points, and strategies to address specific focus areas, formatted in markdown with headings and subheadings."
    )

    executive_brief_task = Task(
        description=f"""
        Synthesize all the gathered information into a comprehensive yet concise executive brief for the meeting with {company_name}. Create the following components:

        1. A detailed one-page executive summary including:
           - Clear statement of the meeting objective
           - List of key attendees and their roles
           - Critical background points about {company_name} and relevant industry context
           - Top 3-5 strategic goals for the meeting, aligned with the objective
           - Brief overview of the meeting structure and key topics to be covered

        2. An in-depth list of key talking points, each supported by:
           - Relevant data or statistics
           - Specific examples or case studies
           - Connection to the company's current situation or challenges

        3. Anticipate and prepare for potential questions:
           - List likely questions from attendees based on their roles and the meeting objective
           - Craft thoughtful, data-driven responses to each question
           - Include any supporting information or additional context that might be needed

        4. Strategic recommendations and next steps:
           - Provide 3-5 actionable recommendations based on the analysis
           - Outline clear next steps for implementation or follow-up
           - Suggest timelines or deadlines for key actions
           - Identify potential challenges or roadblocks and propose mitigation strategies

        Ensure the brief is comprehensive yet concise, highly actionable, and precisely aligned with the meeting objective: {meeting_objective}. The document should be structured for easy navigation and quick reference during the meeting.
        Format your output using markdown with appropriate headings and subheadings.
        """,
        agent=executive_briefing_creator,
        expected_output="A comprehensive executive brief including summary, key talking points, Q&A preparation, and strategic recommendations, formatted in markdown with main headings (H1), section headings (H2), and subsection headings (H3) where appropriate. Use bullet points, numbered lists, and emphasis (bold/italic) for key information."
    )

    # Create the crew
    meeting_prep_crew = Crew(
        agents=[context_analyzer, industry_insights_generator, strategy_formulator, executive_briefing_creator],
        tasks=[context_analysis_task, industry_analysis_task, strategy_development_task, executive_brief_task],
        verbose=True,
        process=Process.sequential
    )

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📝 Meeting Preparation", "🤖 Auto-Joiner"])
    
    with tab1:
        st.header("📝 AI Meeting Preparation")
        
        # Run the crew when the user clicks the button
        if st.button("Prepare Meeting"):
            with st.spinner("AI agents are preparing your meeting..."):
                result = meeting_prep_crew.kickoff()        
            st.markdown(result)
    
    with tab2:
        st.header("🤖 Automated Meeting Joiner")
        st.write("Automatically join Google Meet or Zoom meetings with AI assistance.")
        
        # Meeting URL input
        meeting_url = st.text_input(
            "Meeting URL", 
            placeholder="https://meet.google.com/xxx-xxxx-xxx or https://zoom.us/j/xxxxxxxxx"
        )
        
        # Configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Browser Settings")
            headless_mode = st.checkbox("Headless Mode", value=False, 
                                       help="Run browser in background (no visible window)")
            display_name = st.text_input("Display Name", value="AI Meeting Assistant",
                                       help="Name to show in the meeting")
            wait_timeout = st.slider("Wait Timeout (seconds)", 10, 60, 30,
                                    help="Maximum time to wait for page elements")
        
        with col2:
            st.subheader("🎥 Media Settings")
            auto_mute_camera = st.checkbox("Auto-mute Camera", value=True,
                                         help="Automatically disable camera when joining")
            auto_mute_microphone = st.checkbox("Auto-mute Microphone", value=True,
                                             help="Automatically disable microphone when joining")
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], 
                                    index=1, help="Logging verbosity")
        
        # Authentication Section
        st.subheader("🔐 Authentication & Credentials")
        col_auth1, col_auth2 = st.columns(2)
        
        with col_auth1:
            enable_login = st.checkbox("Enable Auto-Login", value=True,
                                      help="Automatically handle login for Google/Zoom accounts")
            
            if enable_login:
                # Load existing credentials
                try:
                    temp_cred_manager = SecureCredentialManager()
                    stored_platforms = temp_cred_manager.list_stored_platforms()
                    
                    if stored_platforms:
                        st.success(f"✅ Stored credentials: {', '.join(stored_platforms)}")
                    else:
                        st.info("ℹ️ No stored credentials found")
                        
                except Exception as e:
                    st.warning(f"⚠️ Error loading credentials: {str(e)}")
        
        with col_auth2:
            if enable_login:
                with st.expander("🔧 Manage Credentials"):
                    st.markdown("**Add New Credentials:**")
                    
                    platform_choice = st.selectbox("Platform", ["google", "zoom"], key="platform_select")
                    email_input = st.text_input("Email", key="email_input")
                    password_input = st.text_input("Password", type="password", key="password_input")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button("💾 Store Credentials", key="store_creds"):
                            if email_input and password_input:
                                try:
                                    cred_manager = SecureCredentialManager()
                                    cred_manager.store_credentials(platform_choice, email_input, password_input)
                                    st.success(f"✅ Credentials stored for {platform_choice}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to store credentials: {e}")
                            else:
                                st.warning("⚠️ Please enter both email and password")
                    
                    with col_btn2:
                        if st.button("🗑️ Clear All", key="clear_creds"):
                            try:
                                cred_manager = SecureCredentialManager()
                                platforms = cred_manager.list_stored_platforms()
                                for platform in platforms:
                                    cred_manager.remove_credentials(platform)
                                st.success("✅ All credentials cleared")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to clear credentials: {e}")
                    
                    # Session management
                    st.markdown("**Session Management:**")
                    try:
                        session_manager = SessionManager()
                        sessions = session_manager.list_saved_sessions()
                        
                        if sessions:
                            st.markdown("**Saved Sessions:**")
                            for session in sessions:
                                timestamp = session.get('timestamp', 'Unknown')
                                if timestamp != 'Unknown':
                                    try:
                                        parsed_time = datetime.fromisoformat(timestamp)
                                        time_str = parsed_time.strftime("%Y-%m-%d %H:%M")
                                    except:
                                        time_str = timestamp
                                else:
                                    time_str = "Unknown"
                                
                                col_sess1, col_sess2 = st.columns([3, 1])
                                with col_sess1:
                                    st.text(f"📱 {session['platform']} - {session['user_id']} ({time_str})")
                                with col_sess2:
                                    if st.button("🗑️", key=f"clear_session_{session['platform']}_{session['user_id']}"):
                                        session_manager.clear_session(session['platform'], session['user_id'])
                                        st.rerun()
                        else:
                            st.info("No saved sessions found")
                            
                    except Exception as e:
                        st.warning(f"Error loading sessions: {e}")
            else:
                st.markdown("*Auto-login disabled*")
        
        # Audio Capture Section
        st.subheader("🎤 Audio Capture & Transcription")
        col3, col4 = st.columns(2)
        
        with col3:
            enable_audio_capture = st.checkbox("Enable Audio Capture", value=False,
                                             help="Capture meeting audio for transcription")
            audio_capture_mode = st.selectbox("Capture Mode", 
                                            ["system", "browser"], 
                                            index=0,
                                            help="System: capture all audio, Browser: capture only browser audio")
            whisper_model = st.selectbox("Whisper Model", 
                                       ["tiny", "base", "small", "medium", "large"], 
                                       index=1,
                                       help="Larger models are more accurate but slower")
        
        with col4:
            real_time_transcription = st.checkbox("Real-time Transcription", value=False,
                                                help="Enable live transcription during meeting")
            
            if enable_audio_capture:
                st.info("⚠️ **Audio Setup Required**: Click 'Show Setup Instructions' below")
                
                if st.button("🔧 Show Audio Setup Instructions"):
                    system = platform.system().lower()
                    if system == "windows":
                        st.markdown("""
                        **Windows Audio Setup:**
                        1. Enable 'Stereo Mix' in Sound Settings:
                           - Right-click sound icon → Sounds → Recording tab
                           - Right-click empty space → Show Disabled Devices
                           - Enable 'Stereo Mix' device
                        
                        **Alternative: VB-Audio Cable**
                        - Download: https://vb-audio.com/Cable/
                        - Set as default playback device
                        """)
                    elif system == "darwin":
                        st.markdown("""
                        **macOS Audio Setup (BlackHole):**
                        1. Download: https://github.com/ExistentialAudio/BlackHole
                        2. Install the 2ch version
                        3. Create Multi-Output Device in Audio MIDI Setup:
                           - Open Audio MIDI Setup app
                           - Create Multi-Output Device
                           - Select both built-in output and BlackHole 2ch
                        4. Set Multi-Output Device as system output
                        """)
                    elif system == "linux":
                        st.markdown("""
                        **Linux Audio Setup:**
                        1. Install pulseaudio-utils:
                           ```sudo apt-get install pulseaudio-utils```
                        2. List sources: ```pactl list sources short```
                        3. Use monitor source for system audio
                        """)
            else:
                st.markdown("*Audio capture disabled*")
        
        # Advanced options (collapsible)
        with st.expander("🔧 Advanced Options"):
            browser_profile = st.text_input(
                "Browser Profile Path (Optional)", 
                placeholder="/path/to/chrome/profile",
                help="Use existing Chrome profile for saved passwords/settings"
            )
            
            st.info("💡 **Tip**: Using a browser profile can help with saved meeting credentials and preferences.")
        
        # Action buttons
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col1:
            join_button = st.button("🚀 Join Meeting", type="primary")
        
        with col2:
            if ('orchestrator' in st.session_state and st.session_state.orchestrator and 
                st.session_state.orchestrator.is_session_active()) or st.session_state.joiner_session:
                leave_button = st.button("🚪 Leave Meeting", type="secondary")
            else:
                leave_button = False
        
        with col3:
            status_button = st.button("📊 Check Status")
        
        with col4:
            if ('orchestrator' in st.session_state and st.session_state.orchestrator and 
                st.session_state.orchestrator_state in ["error", "ended"]):
                reconnect_button = st.button("🔄 Reconnect", help="Force reconnection to meeting")
            else:
                reconnect_button = False
        
        with col5:
            if 'joiner_session' in st.session_state and st.session_state.joiner_session is not None and hasattr(st.session_state.joiner_session, 'audio_manager') and st.session_state.joiner_session.audio_manager:
                if st.session_state.joiner_session.audio_manager.is_transcribing:
                    transcribe_button = st.button("⏸️ Stop Transcript")
                else:
                    transcribe_button = st.button("▶️ Start Transcript")
            else:
                transcribe_button = False
        
        # Session state management
        if 'orchestrator' not in st.session_state:
            st.session_state.orchestrator = None
        
        if 'joiner_session' not in st.session_state:
            st.session_state.joiner_session = None
        
        if 'meeting_status' not in st.session_state:
            st.session_state.meeting_status = "Not Connected"
        
        if 'orchestrator_state' not in st.session_state:
            st.session_state.orchestrator_state = "idle"
        
        # Handle join meeting with orchestrator
        if join_button and meeting_url:
            if st.session_state.orchestrator is None or not st.session_state.orchestrator.is_session_active():
                try:
                    with st.spinner("🔄 Setting up orchestrated meeting session..."):
                        # Create joiner configuration
                        joiner_config = {
                            'headless': headless_mode,
                            'display_name': display_name,
                            'auto_mute_camera': auto_mute_camera,
                            'auto_mute_microphone': auto_mute_microphone,
                            'wait_timeout': wait_timeout,
                            'log_level': log_level,
                            'browser_profile_path': browser_profile if browser_profile else None,
                            'enable_audio_capture': enable_audio_capture,
                            'audio_capture_mode': audio_capture_mode,
                            'whisper_model': whisper_model,
                            'real_time_transcription': real_time_transcription
                        }
                        
                        # Create orchestrator
                        orchestrator = MeetingOrchestrator(
                            joiner_config=joiner_config,
                            max_join_attempts=3,
                            join_confirmation_timeout=60,
                            meeting_health_check_interval=30,
                            audio_start_delay=5,
                            enable_login=enable_login
                        )
                        
                        # Add state change callback to update UI
                        def update_ui_state(old_state, new_state, details):
                            st.session_state.orchestrator_state = new_state
                            if new_state == "authenticating":
                                st.session_state.meeting_status = "Authenticating..."
                            elif new_state == "joined":
                                st.session_state.meeting_status = "Connected"
                                st.session_state.joiner_session = orchestrator.joiner
                            elif new_state == "recording":
                                st.session_state.meeting_status = "Recording Audio"
                            elif new_state == "transcribing":
                                st.session_state.meeting_status = "Live Transcription"
                            elif new_state == "error":
                                st.session_state.meeting_status = "Error"
                            elif new_state == "ended":
                                st.session_state.meeting_status = "Meeting Ended"
                        
                        orchestrator.add_state_callback("authenticating", update_ui_state)
                        orchestrator.add_state_callback("joined", update_ui_state)
                        orchestrator.add_state_callback("recording", update_ui_state)
                        orchestrator.add_state_callback("transcribing", update_ui_state)
                        orchestrator.add_state_callback("error", update_ui_state)
                        orchestrator.add_state_callback("ended", update_ui_state)
                        
                        # Start orchestrated session
                        success = orchestrator.start_meeting_session(meeting_url)
                        
                        if success:
                            st.session_state.orchestrator = orchestrator
                            st.session_state.orchestrator_state = orchestrator.get_current_state()
                            st.success("🚀 Orchestrated meeting session started!")
                            st.info("⏳ The system will automatically join, start audio capture, and begin transcription...")
                            st.balloons()
                        else:
                            st.error("❌ Failed to start meeting session.")
                            
                except Exception as e:
                    st.error(f"❌ Error starting meeting session: {str(e)}")
                    st.session_state.meeting_status = "Error"
            else:
                st.warning("⚠️ Meeting session already active. Please stop current session first.")
        
        # Handle leave meeting
        if leave_button:
            try:
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.stop_session()
                    st.session_state.orchestrator = None
                
                if st.session_state.joiner_session:
                    try:
                        st.session_state.joiner_session.leave_meeting()
                    except:
                        pass
                    st.session_state.joiner_session = None
                
                st.session_state.meeting_status = "Disconnected"
                st.session_state.orchestrator_state = "idle"
                st.success("👋 Left the meeting successfully!")
            except Exception as e:
                st.error(f"❌ Error leaving meeting: {str(e)}")
        
        # Handle status check
        if status_button:
            try:
                if st.session_state.orchestrator:
                    # Get orchestrator stats
                    stats = st.session_state.orchestrator.get_session_stats()
                    st.session_state.orchestrator_state = stats["current_state"]
                    
                    # Update meeting status based on orchestrator state
                    state_mapping = {
                        "idle": "Not Connected",
                        "authenticating": "Authenticating...",
                        "joining": "Connecting...",
                        "joined": "Connected", 
                        "recording": "Recording Audio",
                        "transcribing": "Live Transcription",
                        "ended": "Meeting Ended",
                        "error": "Error"
                    }
                    st.session_state.meeting_status = state_mapping.get(stats["current_state"], "Unknown")
                    
                    # Display detailed status
                    st.success(f"📊 **Orchestrator Status**: {stats['current_state']}")
                    st.info(f"🔄 **Join Attempts**: {stats['join_attempts']}")
                    if stats['session_duration'] > 0:
                        st.info(f"⏱️ **Session Duration**: {stats['session_duration']:.1f}s")
                    
                elif st.session_state.joiner_session:
                    # Fallback to basic joiner status
                    is_in_meeting = st.session_state.joiner_session.is_in_meeting()
                    if is_in_meeting:
                        st.success("✅ Currently in meeting")
                        st.session_state.meeting_status = "In Meeting"
                    else:
                        st.warning("⚠️ Not currently in meeting")
                        st.session_state.meeting_status = "Connection Lost"
                else:
                    st.warning("⚠️ No active session")
                    
            except Exception as e:
                st.error(f"❌ Error checking status: {str(e)}")
                st.session_state.meeting_status = "Error"
        
        # Handle reconnect
        if reconnect_button and st.session_state.orchestrator:
            try:
                with st.spinner("🔄 Attempting to reconnect..."):
                    success = st.session_state.orchestrator.force_reconnect()
                    if success:
                        st.success("🔄 Reconnection initiated!")
                        st.session_state.orchestrator_state = "joining"
                    else:
                        st.error("❌ Failed to initiate reconnection")
            except Exception as e:
                st.error(f"❌ Error during reconnection: {str(e)}")
        
        # Handle transcription toggle
        if transcribe_button and st.session_state.joiner_session:
            try:
                if st.session_state.joiner_session.audio_manager.is_transcribing:
                    st.session_state.joiner_session.stop_transcription()
                    st.success("⏸️ Transcription stopped")
                else:
                    success = st.session_state.joiner_session.start_transcription()
                    if success:
                        st.success("▶️ Transcription started")
                    else:
                        st.error("❌ Failed to start transcription")
            except Exception as e:
                st.error(f"❌ Error controlling transcription: {str(e)}")
        
        # Display current status
        status_color = {
            "Not Connected": "🔴",
            "Connected": "🟢", 
            "In Meeting": "🟢",
            "Disconnected": "🟡",
            "Connection Lost": "🟠",
            "Error": "🔴"
        }
        
        st.markdown(f"**Status**: {status_color.get(st.session_state.meeting_status, '🔴')} {st.session_state.meeting_status}")
        
        # Orchestrator Status Display
        if st.session_state.orchestrator and st.session_state.orchestrator.is_session_active():
            st.markdown("---")
            st.subheader("🎭 Meeting Orchestrator Status")
            
            # Get current stats
            try:
                stats = st.session_state.orchestrator.get_session_stats()
                
                # Main status row
                col_orch1, col_orch2, col_orch3, col_orch4 = st.columns(4)
                
                with col_orch1:
                    state_emoji = {
                        "idle": "⚫",
                        "authenticating": "🔐",
                        "joining": "🔄", 
                        "joined": "✅",
                        "recording": "🔴",
                        "transcribing": "🟢",
                        "ended": "🏁",
                        "error": "❌"
                    }
                    current_emoji = state_emoji.get(stats["current_state"], "❓")
                    st.markdown(f"**State**: {current_emoji} {stats['current_state']}")
                
                with col_orch2:
                    st.markdown(f"**Attempts**: {stats['join_attempts']}")
                
                with col_orch3:
                    if stats['session_duration'] > 0:
                        duration_str = f"{stats['session_duration']:.0f}s"
                        if stats['session_duration'] > 60:
                            minutes = int(stats['session_duration'] // 60)
                            seconds = int(stats['session_duration'] % 60)
                            duration_str = f"{minutes}m {seconds}s"
                        st.markdown(f"**Duration**: {duration_str}")
                    else:
                        st.markdown(f"**Duration**: --")
                
                with col_orch4:
                    if stats['audio_active']:
                        audio_status = "🔴 Recording"
                    else:
                        audio_status = "⚫ Inactive"
                    st.markdown(f"**Audio**: {audio_status}")
                
                # State history (collapsible)
                if stats['state_history']:
                    with st.expander("🕐 Recent State History"):
                        for i, state_change in enumerate(reversed(stats['state_history'][-5:])):
                            timestamp = state_change['timestamp'].strftime("%H:%M:%S")
                            from_state = state_change['from_state']
                            to_state = state_change['to_state']
                            details = state_change.get('details', '')
                            
                            from_emoji = state_emoji.get(from_state, "❓")
                            to_emoji = state_emoji.get(to_state, "❓")
                            
                            st.markdown(f"`{timestamp}` {from_emoji} {from_state} → {to_emoji} {to_state}")
                            if details:
                                st.markdown(f"    _{details}_")
                
            except Exception as e:
                st.error(f"Error displaying orchestrator status: {e}")
        
        # Live Transcription Display
        if (st.session_state.joiner_session and 
            hasattr(st.session_state.joiner_session, 'audio_manager') and 
            st.session_state.joiner_session.audio_manager and 
            st.session_state.joiner_session.audio_manager.is_recording):
            
            st.markdown("---")
            st.subheader("📝 Live Meeting Transcription")
            
            # Audio status
            audio_manager = st.session_state.joiner_session.audio_manager
            audio_status = "🔴 Recording" if audio_manager.is_recording else "⚫ Stopped"
            transcription_status = "🟢 Transcribing" if audio_manager.is_transcribing else "🟡 Audio Only"
            
            col_audio1, col_audio2, col_audio3 = st.columns(3)
            with col_audio1:
                st.markdown(f"**Audio**: {audio_status}")
            with col_audio2:
                st.markdown(f"**Transcription**: {transcription_status}")
            with col_audio3:
                transcript_count = len(audio_manager.transcripts)
                st.markdown(f"**Segments**: {transcript_count}")
            
            # Live transcript display
            if audio_manager.is_transcribing and audio_manager.transcripts:
                latest_transcripts = audio_manager.get_latest_transcripts(5)
                
                st.markdown("**Recent Transcription:**")
                transcript_container = st.container()
                with transcript_container:
                    for transcript in latest_transcripts:
                        timestamp = transcript['timestamp'].strftime("%H:%M:%S")
                        text = transcript['text']
                        confidence = transcript.get('confidence', 0)
                        
                        # Color code by confidence
                        if confidence > 0.8:
                            confidence_color = "🟢"
                        elif confidence > 0.6:
                            confidence_color = "🟡"
                        else:
                            confidence_color = "🔴"
                        
                        st.markdown(f"`{timestamp}` {confidence_color} {text}")
                
                # Auto-refresh every 5 seconds when transcribing
                if audio_manager.is_transcribing:
                    time.sleep(0.1)  # Small delay to prevent too frequent updates
                    st.rerun()
            
            # Export options
            st.markdown("**Export Options:**")
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                if st.button("💾 Save Audio"):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        audio_file = f"meeting_audio_{timestamp}.wav"
                        success = st.session_state.joiner_session.save_meeting_audio(audio_file)
                        if success:
                            st.success(f"Audio saved: {audio_file}")
                        else:
                            st.error("Failed to save audio")
                    except Exception as e:
                        st.error(f"Error saving audio: {e}")
            
            with col_export2:
                if st.button("📄 Export Transcript"):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        transcript_file = f"meeting_transcript_{timestamp}.txt"
                        success = st.session_state.joiner_session.export_meeting_transcript(transcript_file, "txt")
                        if success:
                            st.success(f"Transcript saved: {transcript_file}")
                        else:
                            st.error("Failed to save transcript")
                    except Exception as e:
                        st.error(f"Error saving transcript: {e}")
            
            with col_export3:
                if st.button("📊 Export JSON"):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        json_file = f"meeting_data_{timestamp}.json"
                        success = st.session_state.joiner_session.export_meeting_transcript(json_file, "json")
                        if success:
                            st.success(f"Data saved: {json_file}")
                        else:
                            st.error("Failed to save data")
                    except Exception as e:
                        st.error(f"Error saving data: {e}")
            
            # Full transcript display (collapsible)
            if audio_manager.transcripts:
                with st.expander("📋 View Full Transcript"):
                    full_transcript = audio_manager.get_full_transcript()
                    if full_transcript:
                        st.text_area("Complete Meeting Transcript", 
                                   value=full_transcript, 
                                   height=300, 
                                   help="Complete transcript of the meeting")
                    else:
                        st.info("No transcript available yet")
        
        # Instructions
        st.markdown("---")
        st.subheader("📚 How to Use Auto-Joiner")
        st.markdown("""
        ### 🎭 Orchestrated Meeting Join:
        1. **Setup Authentication**: Store credentials and enable auto-login
        2. **Enter Meeting URL**: Paste your Google Meet or Zoom meeting link
        3. **Configure Settings**: Adjust browser, media, and audio preferences  
        4. **Start Session**: Click "Join Meeting" to begin orchestrated automation
        5. **Monitor Progress**: Watch real-time state transitions and health checks
        6. **Use Controls**: Reconnect if needed, toggle transcription, check status
        7. **End Session**: Use "Leave Meeting" for complete cleanup
        
        ### 🔄 Smart Orchestration Process:
        1. **Authentication**: Automatic login using stored credentials and session cookies
        2. **Join Attempts**: Multiple retry attempts with exponential backoff
        3. **Join Confirmation**: Waits for actual meeting indicators before proceeding
        4. **Audio Capture**: Delayed start after meeting stabilizes (5s delay)
        5. **Transcription**: Automatic startup if enabled, with audio validation
        6. **Health Monitoring**: Continuous monitoring with automatic recovery
        7. **Lifecycle Management**: Handles meeting end detection and cleanup
        
        ### 🔐 Authentication Features:
        1. **Secure Storage**: Encrypted credential storage using system keyring
        2. **Session Persistence**: Browser cookies and localStorage preservation
        3. **Platform Support**: Google and Zoom OAuth/form-based login
        4. **Session Management**: Automatic session restoration (7-day expiry)
        5. **Credential Management**: Easy storage, viewing, and deletion of credentials
        6. **Privacy**: Local encryption with Fernet, no cloud storage
        
        ### 🎤 Audio Capture & Transcription:
        1. **Enable Audio Capture**: Check the "Enable Audio Capture" option
        2. **Setup Audio**: Follow platform-specific setup instructions
        3. **Choose Capture Mode**: System (all audio) or Browser (meeting only)
        4. **Select Whisper Model**: Balance between speed (tiny/base) and accuracy (large)
        5. **Real-time Transcription**: Enable for live meeting transcription
        6. **Export Options**: Save audio files and transcripts after meetings
        
        **Supported Platforms:**
        - ✅ Google Meet (meet.google.com)
        - ✅ Zoom (zoom.us, zoom.com)
        
        **Meeting Features:**
        - 🤖 Automatic join button detection
        - 🎥 Smart camera/microphone control
        - 🔧 Configurable browser settings
        - 📊 Real-time meeting status monitoring
        
        **Audio Features:**
        - 🎤 Cross-platform audio capture (Windows/Mac/Linux)
        - 🗣️ Real-time speech transcription with Whisper
        - 💾 Audio recording in WAV format
        - 📄 Export transcripts in TXT, JSON, or SRT formats
        - 🔄 Live transcription display with confidence scores
        - ⚙️ Configurable Whisper models for different accuracy/speed needs
        
        **Requirements for Audio Capture:**
        - FFMPEG installed on your system
        - Virtual audio device setup (platform-specific)
        - Whisper model download (automatic on first use)
        
        **Authentication Setup:**
        - Store credentials securely in the app OR
        - Create a .env file with GOOGLE_EMAIL, GOOGLE_PASSWORD, ZOOM_EMAIL, ZOOM_PASSWORD
        - Credentials are encrypted and stored locally
        - Sessions persist for 7 days to minimize repeated logins
        """)

    st.sidebar.markdown("""
    ## How to use this app:
    
    ### 📝 Meeting Preparation:
    1. Enter your API keys in the sidebar.
    2. Provide the requested information about the meeting.
    3. Click 'Prepare Meeting' to generate your comprehensive meeting preparation package.

    The AI agents will work together to:
    - Analyze the meeting context and company background
    - Provide industry insights and trends
    - Develop a tailored meeting strategy and agenda
    - Create an executive brief with key talking points

    ### 🤖 Auto-Joiner with Orchestration:
    1. Switch to the "Auto-Joiner" tab
    2. Enter your meeting URL (Google Meet or Zoom)
    3. Configure browser, media, and audio capture settings
    4. Click "Join Meeting" to start orchestrated session
    
    **Smart Orchestration Features:**
    - 🔄 Automatic retry logic with exponential backoff
    - ✅ Join confirmation verification
    - 🎤 Delayed audio capture after meeting stabilizes
    - 🗣️ Automatic transcription startup
    - 📊 Real-time health monitoring
    - 🔄 Force reconnection capability
    
    **Workflow**: Prepare meeting materials → Start orchestrated session → Monitor progress → Export audio/transcripts
    
    This process may take a few minutes. Please be patient!
    """)
else:
    st.warning("Please enter all API keys in the sidebar before proceeding.")