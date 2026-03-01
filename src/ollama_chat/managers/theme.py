"""Theme and styling management.

Handles Textual theme system, custom themes, persistence, and widget styling.
Enhanced to support full Textual theme switching capabilities.

Integration required in app.py:
- Replace _apply_theme() method
- Replace _style_bubble() method
- Replace _restyle_rendered_bubbles() method
- Use manager for theme switching
- Add theme persistence and keybindings
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from platformdirs import user_config_path

if TYPE_CHECKING:
    from textual.app import App
    from textual.theme import Theme

LOGGER = logging.getLogger(__name__)


class ThemeManager:
    """Manages application theme, styling, and theme switching.

    Responsibilities:
    - Apply Textual themes and custom themes
    - Handle theme persistence
    - Style message bubbles
    - Refresh widget styles
    - Provide theme switching interface
    - Manage custom theme registration
    """

    def __init__(self, config: dict, app_name: str = "ollamaterm", app_author: str = "Web-Dev-Codi") -> None:
        self.config = config
        self.ui_config = config.get("ui", {})
        self.theme_config = config.get("theme", {})
        self.app_name = app_name
        self.app_author = app_author
        
        # Theme state
        self._current_theme_name: str = self.theme_config.get("name", "textual-dark")
        self._using_textual_theme = True
        self._custom_themes: dict[str, Theme] = {}
        
        # Persistence setup
        self._config_path = user_config_path(app_name, app_author, ensure_exists=True) / "theme_settings.json"
        
        # Load persisted theme if enabled
        if self.theme_config.get("persist", True):
            self._load_persisted_theme()
    
    def _load_persisted_theme(self) -> None:
        """Load theme preference from persistent storage."""
        try:
            if self._config_path.exists():
                data = json.loads(self._config_path.read_text())
                saved_theme = data.get("theme")
                if saved_theme:
                    self._current_theme_name = saved_theme
                    LOGGER.info(f"Loaded persisted theme: {saved_theme}")
        except Exception as e:
            LOGGER.warning(f"Failed to load persisted theme: {e}")
    
    def _persist_theme(self, theme_name: str) -> None:
        """Save theme preference to persistent storage."""
        if not self.theme_config.get("persist", True):
            return
            
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"theme": theme_name}
            self._config_path.write_text(json.dumps(data, indent=2))
            LOGGER.info(f"Persisted theme: {theme_name}")
        except Exception as e:
            LOGGER.warning(f"Failed to persist theme: {e}")

    @property
    def current_theme_name(self) -> str:
        """Get the current theme name."""
        return self._current_theme_name
    
    @property
    def is_using_textual_theme(self) -> bool:
        """Check if currently using a Textual theme (vs custom colors)."""
        return self._using_textual_theme and self._current_theme_name != "custom"
    
    def get_available_themes(self, app: App) -> dict[str, Theme]:
        """Get all available themes (built-in + custom).
        
        Args:
            app: Textual app instance
            
        Returns:
            Dictionary mapping theme names to Theme objects
        """
        themes = dict(app.available_themes)
        themes.update(self._custom_themes)
        
        # Add "custom" option for legacy color config
        if "custom" not in themes:
            from textual.theme import Theme
            custom_theme = Theme(
                name="custom",
                primary=self.ui_config.get("user_message_color", "#7aa2f7"),
                background=self.ui_config.get("background_color", "#1a1b26"),
                surface=self.ui_config.get("border_color", "#565f89"),
                dark=True,
            )
            themes["custom"] = custom_theme
            
        return themes
    
    def register_custom_theme(self, theme: Theme) -> None:
        """Register a custom theme.
        
        Args:
            theme: Theme object to register
        """
        self._custom_themes[theme.name] = theme
        LOGGER.info(f"Registered custom theme: {theme.name}")
    
    def register_custom_themes_from_config(self) -> None:
        """Register custom themes defined in configuration."""
        custom_config = self.theme_config.get("custom", {})
        if not isinstance(custom_config, dict):
            return
            
        for theme_name, theme_data in custom_config.items():
            if not isinstance(theme_data, dict):
                continue
                
            try:
                from textual.theme import Theme
                theme = Theme(
                    name=theme_name,
                    primary=theme_data.get("primary", "#7aa2f7"),
                    secondary=theme_data.get("secondary", "#74C7EC"),
                    accent=theme_data.get("accent", "#F5C2E7"),
                    foreground=theme_data.get("foreground", "#CDD6F4"),
                    background=theme_data.get("background", "#1E1E2E"),
                    surface=theme_data.get("surface", "#313244"),
                    panel=theme_data.get("panel", "#45475A"),
                    success=theme_data.get("success", "#A6E3A1"),
                    warning=theme_data.get("warning", "#F9E2AF"),
                    error=theme_data.get("error", "#F38BA8"),
                    dark=theme_data.get("dark", True),
                )
                self.register_custom_theme(theme)
            except Exception as e:
                LOGGER.warning(f"Failed to register custom theme '{theme_name}': {e}")

    def apply_to_bubble(self, bubble, role: str) -> None:
        """Apply theme styling to a message bubble.

        Args:
            bubble: MessageBubble widget
            role: Message role ("user" or "assistant")
        """
        if self.is_using_textual_theme:
            # Let Textual theme handle it through CSS variables
            return

        # Apply custom colors for "custom" theme
        if role == "user":
            bubble.styles.background = self.ui_config.get(
                "user_message_color", "#2a2a2a"
            )
        elif role == "assistant":
            bubble.styles.background = self.ui_config.get(
                "assistant_message_color", "#1a1a1a"
            )

        # Apply border
        border_color = self.ui_config.get("border_color", "#444444")
        if hasattr(bubble, "styles") and hasattr(bubble.styles, "border"):
            bubble.styles.border = ("round", border_color)

    def restyle_all_bubbles(self, bubbles: list) -> None:
        """Refresh styling on all message bubbles.

        Args:
            bubbles: List of MessageBubble widgets
        """
        for bubble in bubbles:
            role = getattr(bubble, "role", None)
            if role:
                self.apply_to_bubble(bubble, role)

    def get_background_color(self) -> str:
        """Get application background color.

        Returns:
            Color string (hex or Textual variable)
        """
        if self.is_using_textual_theme:
            return "$background"
        return self.ui_config.get("background_color", "#000000")

    def switch_theme(self, theme_name: str, app: App) -> bool:
        """Switch to a different theme.

        Args:
            theme_name: Theme name to switch to
            app: Textual app instance
            
        Returns:
            True if theme was switched successfully
        """
        available_themes = self.get_available_themes(app)
        
        if theme_name not in available_themes:
            LOGGER.warning(f"Theme '{theme_name}' not available")
            return False
            
        # Update state
        self._current_theme_name = theme_name
        self._using_textual_theme = theme_name != "custom"
        
        # Apply theme to app
        try:
            app.theme = theme_name
            self._persist_theme(theme_name)
            LOGGER.info(f"Switched to theme: {theme_name}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to switch theme '{theme_name}': {e}")
            return False
    
    def initialize_theme(self, app: App) -> None:
        """Initialize theme system and apply current theme.
        
        Args:
            app: Textual app instance
        """
        # Register custom themes from config
        self.register_custom_themes_from_config()
        
        # Apply current theme
        if self._current_theme_name in self.get_available_themes(app):
            app.theme = self._current_theme_name
            LOGGER.info(f"Initialized with theme: {self._current_theme_name}")
        else:
            # Fallback to textual-dark
            app.theme = "textual-dark"
            self._current_theme_name = "textual-dark"
            LOGGER.warning(f"Theme '{self._current_theme_name}' not available, using textual-dark")
    
    def get_theme_info(self, app: App) -> dict[str, any]:
        """Get information about current theme.
        
        Args:
            app: Textual app instance
            
        Returns:
            Dictionary with theme information
        """
        available_themes = self.get_available_themes(app)
        current_theme = available_themes.get(self._current_theme_name)
        
        return {
            "name": self._current_theme_name,
            "is_textual": self.is_using_textual_theme,
            "is_dark": current_theme.dark if current_theme else True,
            "available_count": len(available_themes),
            "custom_count": len(self._custom_themes),
        }
