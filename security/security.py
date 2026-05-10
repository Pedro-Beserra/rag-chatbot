import re
import unicodedata

import bleach

from config import MAX_INPUT_CHARS


def sanitize_input(text: str) -> str:
    """
    Sanitiza a query do usuário antes de qualquer processamento.

    Etapas aplicadas em ordem:
    1. Guard de tipo/vazio
    2. Truncamento por tamanho máximo de caracteres
    3. Remoção de tags HTML (bleach) — previne XSS se o texto
       for eventualmente renderizado em UI
    4. Remoção de caracteres de controle (\\x00-\\x1f exceto \\t\\n)
       — evita comportamentos inesperados em parsers e logs
    5. Normalização Unicode para NFC — previne bypass via
       homóglifos ou sequências decompostas
    6. Collapse de espaços em branco excessivos
    7. Strip final
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. Truncamento preventivo antes de qualquer processamento pesado
    text = text[:MAX_INPUT_CHARS]

    # 2. Remove tags HTML — strip=True descarta o conteúdo das tags removidas
    text = bleach.clean(text, tags=[], attributes={}, strip=True)

    # 3. Remove caracteres de controle (mantém \t e \n para preservar estrutura)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 4. Normalização Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 5. Colapsa sequências de espaços em branco múltiplos
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
