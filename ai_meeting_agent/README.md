# 🤖 AI Meeting Agent with Auto-Joiner

A comprehensive AI-powered meeting preparation and automation system that combines intelligent meeting preparation with automated meeting joining capabilities.

## ✨ Features

### 🧠 AI Meeting Preparation
- **Context Analysis**: Comprehensive company and meeting background research
- **Industry Insights**: In-depth industry analysis and competitive landscape
- **Strategic Planning**: Tailored meeting strategies and detailed agendas
- **Executive Briefings**: Concise, actionable meeting preparation materials

### 🚀 Automated Meeting Joining
- **Multi-Platform Support**: Google Meet and Zoom compatibility
- **Intelligent Automation**: Smart join button detection and media control
- **Privacy-First**: Auto-mute camera/microphone for privacy
- **Robust Error Handling**: Retry logic and comprehensive error management
- **Session Management**: Monitor meeting health and connection status

## 📁 Project Structure

```
ai_meeting_agent/
├── meeting_agent.py              # Main Streamlit app for AI preparation
├── meeting_auto_joiner.py        # Core auto-joiner class
├── meeting_integration_demo.py   # Streamlit integration demo
├── advanced_audio_demo.py        # Advanced features demonstration
├── config.py                     # Configuration management
├── utils.py                      # Utility functions and helpers
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google Chrome or Chromium browser
- API keys for:
  - Anthropic Claude API [[memory:4161942]]
  - Serper API (for web searches)

### Setup Instructions

1. **Clone and navigate to the directory**:
```bash
cd ai_meeting_agent
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional):
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
export SERPER_API_KEY="your_serper_api_key_here"
export AUTO_JOIN_HEADLESS="true"  # for headless browser mode
```

4. **Verify Chrome installation**:
```bash
python advanced_audio_demo.py
# Select option 3 to check environment
```

## 🚀 Quick Start

### Basic Usage

**1. Simple Auto-Joiner**:
```python
from meeting_auto_joiner import MeetingAutoJoiner

# Basic configuration
config = {
    'headless': False,              # Show browser window
    'display_name': 'AI Assistant', # Your display name
    'auto_mute_camera': True,       # Mute camera on join
    'auto_mute_microphone': True,   # Mute microphone on join
}

# Join meeting
with MeetingAutoJoiner(**config) as joiner:
    success = joiner.join_meeting("https://meet.google.com/abc-defg-hij")
    if success:
        print("✅ Joined meeting successfully!")
        # Meeting continues running...
```

**2. Run the Streamlit Integration**:
```bash
streamlit run meeting_integration_demo.py
```

**3. Run Advanced Demo**:
```bash
python advanced_audio_demo.py
```

### Configuration Options

```python
config = {
    'headless': True,                    # Run browser in background
    'browser_profile_path': None,        # Use existing Chrome profile
    'display_name': 'AI Assistant',      # Name shown in meeting
    'auto_mute_camera': True,            # Auto-mute camera
    'auto_mute_microphone': True,        # Auto-mute microphone
    'wait_timeout': 30,                  # Element wait timeout (seconds)
    'log_level': 'INFO'                  # Logging verbosity
}
```

## 🎯 Supported Platforms

### Google Meet
- ✅ Automatic permission handling
- ✅ Smart join button detection
- ✅ Name entry and media controls
- ✅ Meeting status monitoring
- ✅ Camera/microphone management

### Zoom
- ✅ Browser-based joining (no app required)
- ✅ "Join from Browser" automation
- ✅ Name entry automation
- ✅ Join button detection
- ✅ Meeting status monitoring

## 📋 Usage Examples

### 1. Integration with AI Preparation

```python
# First, prepare meeting materials using AI agents
from meeting_agent import meeting_prep_crew

# Then, automatically join the meeting
from meeting_auto_joiner import MeetingAutoJoiner

# Prepare meeting
meeting_materials = meeting_prep_crew.kickoff()

# Join meeting automatically
with MeetingAutoJoiner(display_name="Prepared AI Assistant") as joiner:
    joiner.join_meeting(meeting_url)
```

### 2. Advanced Monitoring

```python
from advanced_audio_demo import AdvancedMeetingBot

with AdvancedMeetingBot() as bot:
    success = bot.join_meeting_with_retry(meeting_url, max_retries=3)
    
    if success:
        # Monitor meeting health
        stats = bot.get_meeting_stats()
        print(f"Meeting duration: {stats['duration_minutes']} minutes")
        print(f"Status: {stats['status']}")
```

### 3. URL Validation

```python
from utils import MeetingURLParser

# Validate meeting URL
is_valid, platform, info = MeetingURLParser.validate_meeting_url(
    "https://meet.google.com/abc-defg-hij"
)

if is_valid:
    print(f"Valid {platform} meeting: {info['meeting_code']}")
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Required |
| `SERPER_API_KEY` | Serper search API key | Required |
| `AUTO_JOIN_HEADLESS` | Run browser in headless mode | `false` |
| `AUTO_JOIN_TIMEOUT` | Element wait timeout | `30` |
| `CHROME_PROFILE_PATH` | Custom Chrome profile path | None |

### Browser Configuration

The system automatically configures Chrome with optimal settings:

- **Media Permissions**: Auto-allow camera/microphone access
- **Security**: Disable automation detection
- **Performance**: Optimized for meeting platforms
- **Privacy**: Block notifications and unnecessary content

## 🔒 Privacy & Security

- **🛡️ Local Processing**: All automation runs locally
- **🔐 No Data Storage**: Meeting URLs and credentials are not stored
- **👁️ Transparent Operation**: Full logging of all actions
- **🚫 No Recording**: Tool only joins meetings, doesn't record

## 🐛 Troubleshooting

### Common Issues

**1. Chrome Not Found**
```bash
# Install Chrome/Chromium
# macOS: brew install --cask google-chrome
# Ubuntu: sudo apt-get install google-chrome-stable
```

**2. Selenium WebDriver Issues**
```bash
# Update webdriver-manager
pip install --upgrade webdriver-manager
```

**3. Meeting Join Failures**
- Verify meeting URL format
- Check internet connection
- Ensure meeting hasn't started yet (for some platforms)
- Try with `headless=False` to see browser actions

**4. Permission Issues**
```python
# Try with explicit permissions
config = {
    'headless': False,
    'browser_profile_path': '/path/to/chrome/profile'  # Use existing profile
}
```

### Debug Mode

Enable detailed logging:
```python
config = {
    'log_level': 'DEBUG',
    'headless': False  # See browser actions
}
```

## 🔄 Advanced Features

### Session Management
- Automatic session tracking
- Meeting health monitoring
- Reconnection logic
- Session cleanup

### Retry Logic
- Intelligent retry with exponential backoff
- Platform-specific retry strategies
- Comprehensive error handling

### URL Analysis
- Automatic platform detection
- Meeting ID/code extraction
- URL validation and parsing

## 📝 Examples

### Example 1: Scheduled Meeting Join
```python
import schedule
import time
from meeting_auto_joiner import MeetingAutoJoiner

def join_daily_standup():
    with MeetingAutoJoiner(headless=True) as joiner:
        joiner.join_meeting("https://meet.google.com/daily-standup")

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(join_daily_standup)

while True:
    schedule.run_pending()
    time.sleep(1)
```

### Example 2: Multi-Platform Support
```python
meeting_urls = [
    "https://meet.google.com/team-sync",
    "https://zoom.us/j/1234567890",
]

for url in meeting_urls:
    with MeetingAutoJoiner() as joiner:
        success = joiner.join_meeting(url)
        print(f"Joined {url}: {'✅' if success else '❌'}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Legal Notice

- Ensure you have permission to use automated tools for your meetings
- Comply with your organization's automation policies
- Respect meeting platform terms of service
- Use responsibly and ethically

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Enable debug logging
4. Create an issue with detailed information

---

**Made with ❤️ for seamless meeting automation**