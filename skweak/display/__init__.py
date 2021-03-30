"""
spaCy's built in visualization suite for dependencies and named entities.

DOCS: https://spacy.io/api/top-level#displacy
USAGE: https://spacy.io/usage/visualizers
"""
from typing import Union, Iterable, Optional, Dict, Any, Callable
import warnings

from .render import EntityRenderer
from spacy.tokens import Doc, Span
from spacy.errors import Errors, Warnings
from spacy.util import is_in_jupyter


_html = {}
RENDER_WRAPPER = None


def render(
    docs: Union[Iterable[Union[Doc, Span]], Doc, Span],
    style: str = "ent",
    page: bool = False,
    minify: bool = False,
    jupyter: Optional[bool] = None,
    options: Dict[str, Any] = {},
    manual: bool = False,
) -> str:
    """Render displaCy visualisation.

    docs (Union[Iterable[Doc], Doc]): Document(s) to visualise.
    style (str): Visualisation style, 'dep' or 'ent'.
    page (bool): Render markup as full HTML page.
    minify (bool): Minify HTML markup.
    jupyter (bool): Override Jupyter auto-detection.
    options (dict): Visualiser-specific options, e.g. colors.
    manual (bool): Don't parse `Doc` and instead expect a dict/list of dicts.
    RETURNS (str): Rendered HTML markup.

    DOCS: https://spacy.io/api/top-level#displacy.render
    USAGE: https://spacy.io/usage/visualizers
    """
    factories = {
        "ent": (EntityRenderer, parse_ents),
    }
    if style not in factories:
        raise ValueError(Errors.E087.format(style=style))
    if isinstance(docs, (Doc, Span, dict)):
        docs = [docs]
    docs = [obj if not isinstance(obj, Span) else obj.as_doc() for obj in docs]
    if not all(isinstance(obj, (Doc, Span, dict)) for obj in docs):
        raise ValueError(Errors.E096)
    renderer_func, converter = factories[style]
    renderer = renderer_func(options=options)
    parsed = [converter(doc, options) for doc in docs] if not manual else docs
    _html["parsed"] = renderer.render(parsed, page=page, minify=minify).strip()
    html = _html["parsed"]
    if RENDER_WRAPPER is not None:
        html = RENDER_WRAPPER(html)
    if jupyter or (jupyter is None and is_in_jupyter()):
        # return HTML rendered by IPython display()
        # See #4840 for details on span wrapper to disable mathjax
        from IPython.core.display import display, HTML

        return display(HTML('<span class="tex2jax_ignore">{}</span>'.format(html)))
    return html


def serve(
    docs: Union[Iterable[Doc], Doc],
    style: str = "dep",
    page: bool = True,
    minify: bool = False,
    options: Dict[str, Any] = {},
    manual: bool = False,
    port: int = 5000,
    host: str = "0.0.0.0",
) -> None:
    """Serve displaCy visualisation.

    docs (list or Doc): Document(s) to visualise.
    style (str): Visualisation style, 'dep' or 'ent'.
    page (bool): Render markup as full HTML page.
    minify (bool): Minify HTML markup.
    options (dict): Visualiser-specific options, e.g. colors.
    manual (bool): Don't parse `Doc` and instead expect a dict/list of dicts.
    port (int): Port to serve visualisation.
    host (str): Host to serve visualisation.

    DOCS: https://spacy.io/api/top-level#displacy.serve
    USAGE: https://spacy.io/usage/visualizers
    """
    from wsgiref import simple_server

    if is_in_jupyter():
        warnings.warn(Warnings.W011)
    render(docs, style=style, page=page, minify=minify, options=options, manual=manual)
    httpd = simple_server.make_server(host, port, app)
    print(f"\nUsing the '{style}' visualizer")
    print(f"Serving on http://{host}:{port} ...\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"Shutting down server on port {port}.")
    finally:
        httpd.server_close()


def app(environ, start_response):
    headers = [("Content-type", "text/html; charset=utf-8")]
    start_response("200 OK", headers)
    res = _html["parsed"].encode(encoding="utf-8")
    return [res]


def parse_ents(doc: Doc, options: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Generate named entities in [{start: i, end: i, label: 'label'}] format.

    doc (Doc): Document do parse.
    RETURNS (dict): Generated entities keyed by text (original text) and ents.
    """
    ents = [
        {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
        for ent in doc.ents
    ]
    if not ents:
        warnings.warn(Warnings.W006)
    title = doc.user_data.get("title", None) if hasattr(doc, "user_data") else None
    settings = get_doc_settings(doc)
    return {"text": doc.text, "ents": ents, "title": title, "settings": settings}


def set_render_wrapper(func: Callable[[str], str]) -> None:
    """Set an optional wrapper function that is called around the generated
    HTML markup on displacy.render. This can be used to allow integration into
    other platforms, similar to Jupyter Notebooks that require functions to be
    called around the HTML. It can also be used to implement custom callbacks
    on render, or to embed the visualization in a custom page.

    func (callable): Function to call around markup before rendering it. Needs
        to take one argument, the HTML markup, and should return the desired
        output of displacy.render.
    """
    global RENDER_WRAPPER
    if not hasattr(func, "__call__"):
        raise ValueError(Errors.E110.format(obj=type(func)))
    RENDER_WRAPPER = func


def get_doc_settings(doc: Doc) -> Dict[str, Any]:
    return {
        "lang": doc.lang_,
        "direction": doc.vocab.writing_system.get("direction", "ltr"),
    }
