"""
NML Language Server — main entry point.

Provides diagnostics, completions, hover, semantic tokens, document symbols,
and go-to-definition for .nml files via the Language Server Protocol.
"""

from __future__ import annotations

import logging

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from .diagnostics import get_diagnostics
from .completions import get_completions
from .hover import get_hover
from .semantic_tokens import get_semantic_tokens, TOKEN_TYPES, TOKEN_MODIFIERS
from .symbols import get_document_symbols
from .goto_def import get_definition

logger = logging.getLogger(__name__)

server = LanguageServer("nml-lsp", "v0.1.0")


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    _validate(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
    _validate(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams) -> None:
    _validate(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: LanguageServer, params: types.DidCloseTextDocumentParams) -> None:
    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(uri=params.text_document.uri, diagnostics=[])
    )


def _validate(ls: LanguageServer, uri: str) -> None:
    doc = ls.workspace.get_text_document(uri)
    diagnostics = get_diagnostics(doc.source)
    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


@server.feature(types.TEXT_DOCUMENT_COMPLETION)
def completions(ls: LanguageServer, params: types.CompletionParams) -> types.CompletionList:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return get_completions(doc.source, params.position)


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: types.HoverParams) -> types.Hover | None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return get_hover(doc.source, params.position)


@server.feature(types.TEXT_DOCUMENT_DEFINITION)
def definition(ls: LanguageServer, params: types.DefinitionParams) -> types.Location | list[types.Location] | None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return get_definition(doc.source, params.position, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(ls: LanguageServer, params: types.DocumentSymbolParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return get_document_symbols(doc.source)


@server.feature(
    types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    types.SemanticTokensLegend(token_types=TOKEN_TYPES, token_modifiers=TOKEN_MODIFIERS),
)
def semantic_tokens_full(ls: LanguageServer, params: types.SemanticTokensParams) -> types.SemanticTokens:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return get_semantic_tokens(doc.source)


def main():
    logging.basicConfig(level=logging.INFO)
    server.start_io()


if __name__ == "__main__":
    main()
