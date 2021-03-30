from typing import Dict, Any, List, Optional, Union
import uuid

from .templates import TPL_DEP_SVG, TPL_DEP_WORDS, TPL_DEP_WORDS_LEMMA, TPL_DEP_ARCS
from .templates import TPL_ENT, TPL_ENT_RTL, TPL_FIGURE, TPL_TITLE, TPL_PAGE, TPL_ENT_TOOLTIP
from .templates import TPL_ENTS, TPL_ENTS_TOOLTIP
from spacy.util import minify_html, escape_html, registry
from spacy.errors import Errors

from IPython.core.display import display, HTML
import spacy

DEFAULT_LANG = "en"
DEFAULT_DIR = "ltr"
DEFAULT_ENTITY_COLOR = "#ddd"
DEFAULT_LABEL_COLORS = {
    "ORG": "#7aecec",
    "PRODUCT": "#bfeeb7",
    "GPE": "#feca74",
    "LOC": "#ff9561",
    "PERSON": "#aa9cfc",
    "NORP": "#c887fb",
    "FACILITY": "#9cc9cc",
    "EVENT": "#ffeb80",
    "LAW": "#ff8197",
    "LANGUAGE": "#ff8197",
    "WORK_OF_ART": "#f0d0ff",
    "DATE": "#bfe1d9",
    "TIME": "#bfe1d9",
    "MONEY": "#e4e7d2",
    "QUANTITY": "#e4e7d2",
    "ORDINAL": "#e4e7d2",
    "CARDINAL": "#e4e7d2",
    "PERCENT": "#e4e7d2",
}

class EntityRenderer:
    """Render named entities as HTML."""

    style = "ent"

    def __init__(self, options: Dict[str, Any] = {}) -> None:
        """Initialise dependency renderer.

        options (dict): Visualiser-specific options (colors, ents)
        """
        colors = dict(DEFAULT_LABEL_COLORS)
        user_colors = registry.displacy_colors.get_all()
        for user_color in user_colors.values():
            if callable(user_color):
                # Since this comes from the function registry, we want to make
                # sure we support functions that *return* a dict of colors
                user_color = user_color()
            if not isinstance(user_color, dict):
                raise ValueError(Errors.E925.format(obj=type(user_color)))
            colors.update(user_color)
        colors.update(options.get("colors", {}))
        self.default_color = DEFAULT_ENTITY_COLOR
        self.colors = {label.upper(): color for label, color in colors.items()}
        self.ents = options.get("ents", None)
        if self.ents is not None:
            self.ents = [ent.upper() for ent in self.ents]
        self.direction = DEFAULT_DIR
        self.lang = DEFAULT_LANG
        template = options.get("template")
        if template:
            self.ent_template = template
        else:
            if self.direction == "rtl":
                self.ent_template = TPL_ENT_RTL
            else:
                #self.ent_template = TPL_ENT
                self.ent_template = TPL_ENT_TOOLTIP

    def render(
        self, parsed: List[Dict[str, Any]], page: bool = False, minify: bool = False
    ) -> str:
        """Render complete markup.

        parsed (list): Dependency parses to render.
        page (bool): Render parses wrapped as full HTML page.
        minify (bool): Minify HTML markup.
        RETURNS (str): Rendered HTML markup.
        """
        rendered = []
        for i, p in enumerate(parsed):
            if i == 0:
                settings = p.get("settings", {})
                self.direction = settings.get("direction", DEFAULT_DIR)
                self.lang = settings.get("lang", DEFAULT_LANG)
            rendered.append(self.render_ents(p["text"], p["ents"], p.get("title")))
        if page:
            docs = "".join([TPL_FIGURE.format(content=doc) for doc in rendered])
            markup = TPL_PAGE.format(content=docs, lang=self.lang, dir=self.direction)
        else:
            markup = "".join(rendered)
        if minify:
            return minify_html(markup)
        return markup

    def render_ents(
        self, text: str, spans: List[Dict[str, Any]], title: Optional[str]
    ) -> str:
        """Render entities in text.

        text (str): Original text.
        spans (list): Individual entity spans and their start, end and label.
        title (str / None): Document title set in Doc.user_data['title'].
        """
        markup = """<style>
.ttooltip {
  position: relative;
  display: inline-block;
}
.ttooltip .ttooltiptext {
  visibility: hidden;
  width: 240px;
  background-color: DarkTurquoise;
  color: #fff;
  text-align: center;
  padding: 5px 0;
  border-radius: 6px;

  position: absolute;
  z-index: 1;
}
.ttooltip:hover .ttooltiptext {
  visibility: visible;
}
.txt:hover {
    text-decoration: underline;
}
</style>"""
        offset = 0
        for span in spans:
            label = span["label"]
            start = span["start"]
            end = span["end"]
            additional_params = span.get("params", {})
            entity = escape_html(text[start:end])
            fragments = text[offset:start].split("\n")
            for i, fragment in enumerate(fragments):
                markup += escape_html(fragment)
                if len(fragments) > 1 and i != len(fragments) - 1:
                    markup += "</br>"
            if self.ents is None or label.upper() in self.ents:
                color = self.colors.get(label.upper(), self.default_color)
                ent_settings = {"label": label, "text": entity, "bg": color}
                ent_settings.update(additional_params)
                markup += self.ent_template.format(**ent_settings)
            else:
                markup += entity
            offset = end
        fragments = text[offset:].split("\n")
        for i, fragment in enumerate(fragments):
            markup += escape_html(fragment)
            if len(fragments) > 1 and i != len(fragments) - 1:
                markup += "</br>"
        markup = TPL_ENTS_TOOLTIP.format(content=markup, dir=self.direction, height=1.5)
        if title:
            markup = TPL_TITLE.format(title=title) + markup
        return markup

