import os
import pdfplumber
import spacy
import re
from sent_class import Sentence, Token, generate_conll
from eng_rb_anaphora import resolve_anaphora_en, sentences_to_conll as en_to_conll
from ru_rb_anaphora import resolve_anaphora_ru, sentences_to_conll as ru_to_conll
from romani_rb_anaphora import resolve_anaphora_rom, sentences_to_conll as rom_to_conll
from make_graph_spacy import build_en_graph_spacy, build_ru_graph_spacy, build_rom_graph_spacy
from graph.graph import visualize_graph_interactive
from compare_graphs import compare_graphs
from add_anaphora_edges import add_anaphora_edges
import unicodedata
    

nlp_models = {
    "en": spacy.load("en_core_web_sm"),
    "ru": spacy.load("ru_core_news_sm"),
    "rom": None  # кастомный парсер
}

def parse_romani_text(text):
    def normalize_word(word):
        if not word:
            return ""
        word = word.lower()
        replacements = {
            'Ќ': 'к', 'ќ': 'к', 'Ћ': 'ч', 'ћ': 'ч',
            'Ѓ': 'г', 'ѓ': 'г', 'Ґ': 'г', 'ґ': 'г',
            'Џ': 'ц', 'џ': 'ц', 'ѐ': 'е',
        }
        for old, new in replacements.items():
            word = word.replace(old, new)
        word = unicodedata.normalize('NFD', word)
        word = ''.join(c for c in word if unicodedata.category(c) != 'Mn')
        word = re.sub(r'[^\w]', '', word)
        return word
    
    def is_cyrillic_word(word):
        if not word:
            return False
        cyrillic_chars = sum(1 for c in word if ord(c) >= 0x0400)
        latin_chars = sum(1 for c in word if ord(c) < 0x0400 and c.isalpha())
        return cyrillic_chars > latin_chars
    
    def is_english_phrase(text):
        chars = [c for c in text if c.isalpha()]
        if not chars:
            return False
        latin_ratio = sum(1 for c in chars if ord(c) < 128) / len(chars)
        return latin_ratio > 0.5
    
    sentences = []
    print(f"Длина текста: {len(text)} символов")
    bible_start = re.search(
        r'Англэды́р\s+сарэ́стыр|Сыр\s+Дэвэ́л\s+Создыя́|^1\s+Англэды́р|'
        r'пхэндя́\s+Дэвэ́л|создыя́\s+болыбэ́н',
        text, 
        re.MULTILINE
    )
    if bible_start:
        text = text[bible_start.start():]
    lines = text.split('\n')
    cleaned_lines = []
    english_markers = [
        'bible', 'testament', 'chapter', 'verse', 'isbn', 'copyright',
        'publisher', 'printed', 'translation', 'rights', 'reserved',
        'www.', '.com', '.org', '.net', 'http', 'mail',
        'united', 'societies', 'distribution', 'literature',
        'atlanta', 'england', 'germany', 'switzerland', 'usa',
        'crawley', 'dillenburg', 'herbligen', 'préverenges',
        'handbook', 'dictionary', 'theology', 'exegesis',
        'grand', 'rapids', 'zondervan', 'brill', 'leiden',
    ]
    
    for line in lines:
        line_lower = line.lower()
        
        # Пропускаем строки с английскими маркерами
        if any(marker in line_lower for marker in english_markers):
            continue
        chars = [c for c in line if c.isalpha()]
        if chars:
            latin_ratio = sum(1 for c in chars if ord(c) < 128) / len(chars)
            if latin_ratio > 0.4 and len(line) > 20:
                continue
        
        # Пропускаем строки, состоящие только из цифр и пунктуации
        if re.match(r'^[\d\s\.\-\–\—\.,:;\(\)\[\]\{\}]+$', line.strip()):
            continue
        
        # Пропускаем строки с номерами страниц
        if re.match(r'^\s*\d+\s*$', line):
            continue
        
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    # Удаляем все, что в скобках на английском
    text = re.sub(r'\([A-Za-z][^)]*\)', '', text)
    # Удаляем английские слова, оставшиеся в тексте
    text = re.sub(r'\b[A-Za-z]{3,}\b', '', text)
    verse_pattern = r'(?:^|\s+)(\d+)\s+'
    
    parts = []
    last_end = 0
    
    for match in re.finditer(verse_pattern, text, re.MULTILINE):
        verse_num = match.group(1)
        start = match.start()
        end = match.end()
        
        if start > last_end:
            part_text = text[last_end:start].strip()
            if part_text and len(part_text) > 5 and not is_english_phrase(part_text):
                parts.append((verse_num, part_text))
        
        last_end = end
    
    if last_end < len(text):
        part_text = text[last_end:].strip()
        if part_text and len(part_text) > 5 and not is_english_phrase(part_text):
            parts.append(('?', part_text))
    
    PRONOUNS_NORM = {
        'мэ', 'ме', 'ту', 'ев', 'йов', 'ой', 'амен', 'амэн', 'тумен', 'тумэн', 'вон',
        'ман', 'тут', 'лэс', 'лес', 'ла', 'лэн', 'лен', 'амэ', 'тумэ',
        'миро', 'мири', 'мирэ', 'тиро', 'тири', 'тирэ',
        'лэско', 'лэски', 'лако', 'лаки', 'амаро', 'амари', 'тумаро', 'тумари', 'лэнго', 'лэнги',
        'пэскиро', 'пэскири', 'пэскирэ', 'пэско', 'пэс',
        'адава', 'адая', 'адалэ', 'дава', 'долэ', 'дола', 'кадя', 'када', 'одова', 'одоя', 'одолэ',
        'саво', 'сави', 'савэ', 'со', 'ко', 'кон', 'кай',
    }
    
    VERB_MARKERS_NORM = {
        'дья', 'дя', 'дыя', 'кэрдя', 'кэр', 'пхэндя', 'пхэн',
        'дыкхця', 'дыкх', 'сыс', 'исин', 'исы́н', 'создыя', 'созд',
        'роскэрдя', 'яця', 'яч', 'барьякир', 'скэд', 'джя', 'джал',
        'ав', 'авэл', 'мэк', 'биян', 'бахтякир', 'хулаин',
        'дыя', 'лыя', 'яндя', 'чхудя', 'почхудя', 'приячэл',
        'кхарэл', 'кхарэла', 'пхэрдэ', 'бияндён', 'розджал',
    }
    
    sent_idx = 0
    total_pronouns = 0
    
    for verse_num, part_text in parts:
        if is_english_phrase(part_text):
            continue
        tokens_raw = re.findall(r'[\wЌЋЃҐЏ́ѐ\-]+', part_text)
        cyrillic_tokens = []
        for tok in tokens_raw:
            if len(tok) >= 2:
                has_cyrillic = any(ord(c) >= 0x0400 for c in tok)
                if has_cyrillic:
                    tok_clean = re.sub(r'[A-Za-z]', '', tok)
                    if len(tok_clean) >= 2:
                        cyrillic_tokens.append(tok_clean)
        
        if len(cyrillic_tokens) < 2:
            continue
        
        tokens = []
        current_id = 1
        pronouns_found = []
        
        for tok_form in cyrillic_tokens:
            tok_norm = normalize_word(tok_form)
            if not tok_norm:
                continue
            
            # Определяем часть речи
            if tok_norm in PRONOUNS_NORM:
                pos = "PRON"
                pronouns_found.append(tok_form)
                total_pronouns += 1
            elif tok_norm in VERB_MARKERS_NORM:
                pos = "VERB"
            elif tok_form[0].isupper() and not tok_norm.endswith(('тко', 'ткири', 'но', 'ны', 'ло', 'лы')):
                pos = "PROPN"
            elif tok_norm.endswith(('ибэн', 'ыбэн', 'ипэн', 'ыпэн', 'ибны', 'ыбны')):
                pos = "NOUN"
            elif tok_norm.endswith(('тко', 'ткон', 'ткири', 'итко', 'ытко', 'но', 'ны', 'ло', 'лы', 'ро', 'ры')):
                pos = "ADJ"
            elif tok_norm.endswith(('о', 'и', 'а', 'э', 'я', 'ы')):
                pos = "NOUN"
            else:
                pos = "NOUN"
            
            lemma = tok_norm
            head = 0
            deprel = "dep"
            
            if current_id == 1:
                deprel = "root"
            elif tokens:
                prev = tokens[-1]
                if pos == "VERB" and prev.pos in ('NOUN', 'PROPN', 'PRON'):
                    prev.head = current_id
                    prev.deprel = "nsubj"
                    head = 0
                    deprel = "root"
                elif prev.pos == "VERB" and pos in ('NOUN', 'PROPN', 'PRON'):
                    head = prev.id
                    deprel = "obj"
# Словарь английских лемм, которые spaCy мог присвоить
            ENGLISH_FAKE_LEMMAS = {
                'be', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
                'do', 'does', 'did', 'say', 'says', 'said', 'go', 'goes', 'went',
                'come', 'comes', 'came', 'make', 'makes', 'made', 'take', 'takes', 'took'
            }
            lemma = tok_norm
            if lemma in ENGLISH_FAKE_LEMMAS:
                lemma = tok_norm  # используем нормализованную форму
            token = Token(
                id=current_id,
                form=tok_form,
                lemma=lemma,
                pos=pos,
                xpos="_",
                head=head,
                deprel=deprel
            )
            
            tokens.append(token)
            current_id += 1
        
        if tokens:
            has_root = any(t.deprel == "root" for t in tokens)
            if not has_root:
                for t in tokens:
                    if t.pos == "VERB":
                        t.head = 0
                        t.deprel = "root"
                        has_root = True
                        break
            if not has_root:
                tokens[0].head = 0
                tokens[0].deprel = "root"
            
            sent_obj = Sentence(
                tokens=tokens,
                sent_id=sent_idx,
                text=' '.join(t.form for t in tokens)
            )
            
            sentences.append(sent_obj)
            sent_idx += 1
            
    ENGLISH_STOPWORDS = {
        'be', 'is', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing',
        'say', 'says', 'said', 'saying',
        'go', 'goes', 'went', 'gone', 'going',
        'come', 'comes', 'came', 'coming',
        'make', 'makes', 'made', 'making',
        'take', 'takes', 'took', 'taken', 'taking',
        'get', 'gets', 'got', 'gotten', 'getting',
        'see', 'sees', 'saw', 'seen', 'seeing',
        'know', 'knows', 'knew', 'known', 'knowing',
        'think', 'thinks', 'thought', 'thinking',
        'want', 'wants', 'wanted', 'wanting',
        'like', 'likes', 'liked', 'liking',
        'need', 'needs', 'needed', 'needing',
        'the', 'a', 'an', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for',
        'with', 'by', 'from', 'as', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their',
    }
    
    filtered_sentences = []
    for sent in sentences:
        filtered_tokens = []
        for tok in sent.tokens:
            # Проверяем форму и лемму
            form_lower = tok.form.lower()
            lemma_lower = tok.lemma.lower() if tok.lemma else ""
            # Пропускаем английские стоп-слова
            if form_lower in ENGLISH_STOPWORDS or lemma_lower in ENGLISH_STOPWORDS:
                continue
            # Пропускаем слова, состоящие только из латиницы
            if all(ord(c) < 128 for c in tok.form if c.isalpha()):
                continue
            # Если лемма английская - заменяем на форму
            if tok.lemma and all(ord(c) < 128 for c in tok.lemma if c.isalpha()):
                tok.lemma = tok.form.lower()
            
            filtered_tokens.append(tok)
        if len(filtered_tokens) >= 2:
            # Перенумеруем токены
            for i, tok in enumerate(filtered_tokens, 1):
                tok.id = i
                # Корректируем head ссылки
                if tok.head > len(filtered_tokens):
                    tok.head = 0
            
            sent.tokens = filtered_tokens
            sent.text = ' '.join(t.form for t in filtered_tokens)
            filtered_sentences.append(sent)
    
    sentences = filtered_sentences
    return sentences

def clean_bible_text(text, lang):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Пропускаем технический мусор
        if re.search(r'\.indb|\.pdf|EXE|_\d+|www\.|ISBN|©|All rights reserved|G\.B\.V\.|U\.B\.S\.|Crawley|Dillenburg', line, re.IGNORECASE):
            continue
        if re.match(r'^[\d\s\.:\-]+$', line.strip()):
            continue
        
        # Для русского и цыганского - убираем английские строки
        if lang in ('ru', 'rom'):
            latin_ratio = sum(1 for c in line if c.isalpha() and ord(c) < 128) / max(len(line), 1)
            if latin_ratio > 0.5:
                continue
        
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def pdf_to_text(pdf_path, lang=None, start_page=None, end_page=None):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        # Определяем диапазон страниц
        if start_page is None:
            start_idx = 0
        else:
            start_idx = max(0, start_page - 1)
        
        if end_page is None:
            end_idx = total_pages
        else:
            end_idx = min(total_pages, end_page)
        for page_num in range(start_idx, end_idx):
            page = pdf.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    if lang:
        text = clean_bible_text(text, lang)
    
    return text

def text_to_sentences(text, lang):
    if lang == "rom":
        return parse_romani_text(text)
    
    nlp = nlp_models[lang]
    doc = nlp(text)
    
    sentences = []
    for i, sent in enumerate(doc.sents):
        tokens = []
        for token in sent:
            tok = Token(
                id=token.i - sent.start + 1,
                form=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                xpos=token.tag_,
                head=(token.head.i - sent.start + 1) if token.head in sent else 0,
                deprel=token.dep_
            )
            tokens.append(tok)
        sentences.append(Sentence(tokens=tokens, sent_id=i, text=sent.text))
    
    return sentences

def text_to_conll(text, output_file, lang):
    sentences = text_to_sentences(text, lang)
    generate_conll(sentences, output_file)
    return sentences

def analyze_graph_metrics(graph, name):
    metrics = {
        "name": name,
        "vertices": len(graph.vertices),
        "edges": len(graph.edges),
        "density": 0,
        "edge_types": {},
        "top_vertices": []
    }
    
    if metrics["vertices"] > 0:
        max_edges = metrics["vertices"] * (metrics["vertices"] - 1)
        if max_edges > 0:
            metrics["density"] = metrics["edges"] / max_edges
        
        for edge in graph.edges:
            rel = edge.meaning
            metrics["edge_types"][rel] = metrics["edge_types"].get(rel, 0) + 1
        
        degree = {}
        for edge in graph.edges:
            degree[edge.agent_1] = degree.get(edge.agent_1, 0) + 1
            degree[edge.agent_2] = degree.get(edge.agent_2, 0) + 1
        
        metrics["top_vertices"] = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return metrics

def process_language_with_graph(pdf_path, lang, limit=50000, start_page=None):    
    # Настройки для цыганского
    if lang == "rom":
        if start_page is None:
            start_page = 34  # Пропускаем предисловие на русском/английском
        if limit == 50000:
            limit = 300000
    # Извлекаем текст из PDF
    text = pdf_to_text(pdf_path, lang=lang, start_page=start_page)
    
    if not text:
        return None, None
    
    preview = text[:300].replace('\n', ' ')
    
    # Применяем лимит
    text = text[:limit]
    
    conll_file = f"temp_{lang}.conll"    
    if lang == "en":
        sentences = text_to_sentences(text, lang)
        generate_conll(sentences, conll_file)
        sentences = resolve_anaphora_en(conll_file)
        conll_lines = en_to_conll(sentences)        
        graph = build_en_graph_spacy(sentences)
        anaphora_edges = add_anaphora_edges(graph, sentences, "en")
    
    elif lang == "ru":
        sentences = text_to_sentences(text, lang)
        generate_conll(sentences, conll_file)
        sentences = resolve_anaphora_ru(conll_file)
        conll_lines = ru_to_conll(sentences)        
        graph = build_ru_graph_spacy(sentences)
        anaphora_edges = add_anaphora_edges(graph, sentences, "ru")

    
    elif lang == "rom":
        sentences = parse_romani_text(text)
        generate_conll(sentences, conll_file)
        sentences = resolve_anaphora_rom(conll_file)
        conll_lines = rom_to_conll(sentences)
        
        ENGLISH_STOPWORDS = {
            'be', 'is', 'are', 'was', 'were', 'been', 'being',
            'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing',
            'say', 'says', 'said', 'saying',
            'go', 'goes', 'went', 'gone', 'going',
            'come', 'comes', 'came', 'coming',
            'make', 'makes', 'made', 'making',
            'take', 'takes', 'took', 'taken', 'taking',
            'get', 'gets', 'got', 'gotten', 'getting',
            'see', 'sees', 'saw', 'seen', 'seeing',
            'know', 'knows', 'knew', 'known', 'knowing',
            'think', 'thinks', 'thought', 'thinking',
            'want', 'wants', 'wanted', 'wanting',
            'like', 'likes', 'liked', 'liking',
            'need', 'needs', 'needed', 'needing',
            'the', 'a', 'an', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for',
            'with', 'by', 'from', 'as', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'bible', 'testament', 'chapter', 'verse', 'copyright', 'reserved',
            'rights', 'international', 'publisher', 'printed', 'translation', 'does', 'can'
        }
        
        filtered_sentences = []
        filtered_tokens_count = 0
        total_tokens_count = 0
        
        for sent in sentences:
            total_tokens_count += len(sent.tokens)
            filtered_tokens = []
            
            for tok in sent.tokens:
                form_lower = tok.form.lower()
                lemma_lower = tok.lemma.lower() if tok.lemma else ""
                
                # Пропускаем английские стоп-слова
                if form_lower in ENGLISH_STOPWORDS or lemma_lower in ENGLISH_STOPWORDS:
                    filtered_tokens_count += 1
                    continue
                
                # Пропускаем слова, состоящие только из латиницы
                if all(ord(c) < 128 for c in tok.form if c.isalpha()):
                    filtered_tokens_count += 1
                    continue
                
                # Если лемма английская - заменяем на форму
                if tok.lemma and all(ord(c) < 128 for c in tok.lemma if c.isalpha()):
                    tok.lemma = tok.form.lower()
                
                filtered_tokens.append(tok)
            
            if len(filtered_tokens) >= 2:
                old_to_new = {}
                for i, tok in enumerate(filtered_tokens, 1):
                    old_to_new[tok.id] = i
                    tok.id = i
                
                for tok in filtered_tokens:
                    if tok.head in old_to_new:
                        tok.head = old_to_new[tok.head]
                    else:
                        tok.head = 0
                
                sent.tokens = filtered_tokens
                sent.text = ' '.join(t.form for t in filtered_tokens)
                filtered_sentences.append(sent)
        
        sentences = filtered_sentences
        conll_lines = rom_to_conll(sentences)
        graph = build_rom_graph_spacy(sentences)
        anaphora_edges = add_anaphora_edges(graph, sentences, "rom")

    
    else:
        return None, None
    
    output_file = f"resolved_{lang}.conll"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(conll_lines))
    return sentences, graph

def analyze_graph_quality(graph, sentences, lang):
    if len(graph.vertices) > 0:
        max_edges = len(graph.vertices) * (len(graph.vertices) - 1)
        density = len(graph.edges) / max_edges if max_edges > 0 else 0
        print(f"Плотность: {density:.6f}")
        print(f"Средняя степень вершины: {2 * len(graph.edges) / len(graph.vertices):.2f}")
    # Распределение по длине
    lengths = [len(v) for v in graph.vertices]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    print(f"Средняя длина метки: {avg_length:.1f} символов")
    print(f"Мин/макс длина: {min(lengths) if lengths else 0}/{max(lengths) if lengths else 0}")
    
    # Распределение по частям речи (если есть в sentences)
    pos_dist = {}
    for sent in sentences:
        for tok in sent.tokens:
            pos = tok.pos if tok.pos else "UNK"
            pos_dist[pos] = pos_dist.get(pos, 0) + 1
    
    if pos_dist:
        for pos, count in sorted(pos_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = count / sum(pos_dist.values()) * 100
            print(f"     - {pos}: {count} ({pct:.1f}%)")
    
    degree = {}
    for edge in graph.edges:
        degree[edge.agent_1] = degree.get(edge.agent_1, 0) + 1
        degree[edge.agent_2] = degree.get(edge.agent_2, 0) + 1
    
    top_vertices = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (vertex, deg) in enumerate(top_vertices, 1):
        print(f"   {i:2}. {vertex:<30} степень: {deg}")
    edge_types = {}
    for edge in graph.edges:
        rel = edge.meaning
        edge_types[rel] = edge_types.get(rel, 0) + 1
    
    for rel, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True)[:15]:
        pct = count / len(graph.edges) * 100 if graph.edges else 0
        print(f"   • {rel:<20} {count:4} ({pct:.1f}%)")
    components = get_connected_components(graph)
    if components:
        sizes = sorted([len(c) for c in components], reverse=True)
        isolated = [c for c in components if len(c) == 1]
        if sizes[0] > 1:
            giant_pct = sizes[0] / len(graph.vertices) * 100
    # Проверка на английские слова
    english_in_vertices = []
    for v in graph.vertices:
        if all(ord(c) < 128 for c in v if c.isalpha()):
            english_in_vertices.append(v)
    
    if english_in_vertices:
        for v in english_in_vertices[:10]:
            print(f"     - {v}")
    else:
        print(f"Все вершины на кириллице")
    
    # Проверка на мусорные символы
    garbage_vertices = []
    for v in graph.vertices:
        if re.search(r'[^\w\s\-а-яёЌЋЃҐЏ́]', v, re.IGNORECASE):
            garbage_vertices.append(v)
    
    if garbage_vertices:
        for v in garbage_vertices[:5]:
            print(f"     - {v}")
    else:
        print(f"Нет вершин с мусорными символами")
    
    if lang == "rom":        
        # Ключевые библейские концепты
        biblical_concepts = {
            'дэвэл', 'дэвэ́л', 'рай', 'духо', 'фано',
            'болыбэн', 'пхув', 'дуд', 'свэто', 'калыпэн',
            'дывэс', 'рат', 'пани', 'мори', 'дэрьява',
            'мануш', 'адам', 'ева', 'джувля', 'мурш',
        }
        
        found_concepts = []
        for v in graph.vertices:
            v_clean = v.replace('́', '').lower()
            for concept in biblical_concepts:
                if concept in v_clean:
                    found_concepts.append((v, concept))
                    break
        
        if found_concepts:
            print(f"Найдено библейских концептов: {len(found_concepts)}")
            for v, c in found_concepts[:15]:
                print(f"     - {v} (концепт: {c})")
        
        # Проверка наличия ключевых персонажей
        key_figures = ['дэвэл', 'адам', 'ева', 'каин', 'авель', 'нои', 'авраам']
        found_figures = []
        for v in graph.vertices:
            v_clean = v.replace('́', '').lower()
            for fig in key_figures:
                if fig in v_clean:
                    found_figures.append(v)
                    break
        
        print(f"Ключевых персонажей найдено: {len(set(found_figures))}")
    
    return {
        'vertices': len(graph.vertices),
        'edges': len(graph.edges),
        'density': density if len(graph.vertices) > 0 else 0,
        'components': len(components),
        'isolated': len(isolated) if components else 0,
        'top_vertices': top_vertices[:10],
        'edge_types': edge_types,
        'english_vertices': len(english_in_vertices),
    }


def get_connected_components(graph):
    # Строим список смежности
    adj = {}
    for v in graph.vertices:
        adj[v] = set()
    
    for edge in graph.edges:
        adj[edge.agent_1].add(edge.agent_2)
        adj[edge.agent_2].add(edge.agent_1)
    
    visited = set()
    components = []
    
    for v in graph.vertices:
        if v not in visited:
            # BFS
            component = []
            queue = [v]
            visited.add(v)
            
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
    
    return components


def analyze_vertex_quality_detailed(graph, sentences, output_file=None):
    # Степени вершин
    degree = {}
    for edge in graph.edges:
        degree[edge.agent_1] = degree.get(edge.agent_1, 0) + 1
        degree[edge.agent_2] = degree.get(edge.agent_2, 0) + 1
    
    # Частотность в тексте
    word_freq = {}
    for sent in sentences:
        for tok in sent.tokens:
            word = tok.form.lower().replace('́', '')
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Собираем статистику по всем вершинам
    vertex_stats = []
    for v in graph.vertices:
        v_clean = v.lower().replace('́', '')
        
        # Ищем частотность
        freq = 0
        for word, count in word_freq.items():
            if v_clean in word or word in v_clean:
                freq = max(freq, count)
        
        stats = {
            'vertex': v,
            'length': len(v),
            'degree': degree.get(v, 0),
            'freq_in_text': freq,
            'is_cyrillic': any(ord(c) >= 0x0400 for c in v),
            'has_diacritic': any(c in v for c in '́ЌЋЃҐЏ'),
        }
        vertex_stats.append(stats)
    
    # Сортировка по степени
    vertex_stats.sort(key=lambda x: x['degree'], reverse=True)

    for i, s in enumerate(vertex_stats[:30], 1):
        cyr = "✓" if s['is_cyrillic'] else "✗"
    non_cyrillic = [s for s in vertex_stats if not s['is_cyrillic']]
    
    isolated = [s for s in vertex_stats if s['degree'] == 0]
    if isolated:
        for s in isolated[:10]:
            print(f"     - {s['vertex']}")
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Вершина\tСтепень\tДлина\tЧастота\tКириллица\n")
            for s in vertex_stats:
                f.write(f"{s['vertex']}\t{s['degree']}\t{s['length']}\t{s['freq_in_text']}\t{s['is_cyrillic']}\n")
    
    return vertex_stats

def analyze_all_languages():
    files = {
        "en": "Bible_eng.pdf",
        "ru": "Bible_rus.pdf",
        "rom": "Bible_rom.pdf"
    }
    configs = {
        "en": {"limit": 300000, "start_page": None},
        "ru": {"limit": 300000, "start_page": None},
        "rom": {"limit": 300000, "start_page": 34},  # Начинаем с 34 страницы (Бытие)
    }
    
    graphs = {}
    all_metrics = {}
    sentences_dict = {}
    
    for lang, filename in files.items():
        path = os.path.join(filename)
        
        if not os.path.exists(path):
            continue
        
        config = configs[lang]
        sentences, graph = process_language_with_graph(
            path, lang, 
            limit=config["limit"], 
            start_page=config["start_page"]
        )
        
        if graph:
            graphs[lang] = graph
            sentences_dict[lang] = sentences
            metrics = analyze_graph_metrics(graph, lang)
            all_metrics[lang] = metrics
            quality_metrics = analyze_graph_quality(graph, sentences, lang)
            
            if lang == "rom":
                analyze_vertex_quality_detailed(graph, sentences, "rom_vertex_quality.tsv")
            
            html_file = f"graph_{lang}_bible.html"
            visualize_graph_interactive(graph, output=html_file)
            graphs[lang] = graph
            
        
        if len(graphs) == 3:
            results = compare_graphs(
                graphs['en'], graphs['ru'], graphs['rom'],
                sentences_dict['en'], sentences_dict['ru'], sentences_dict['rom']
            )
    
    if len(all_metrics) > 1:
        for metric_name in ['vertices', 'edges', 'density']:
            print(f"{metric_name.capitalize():<20}", end=" ")
            for lang in ["en", "ru", "rom"]:
                if lang in all_metrics:
                    m = all_metrics[lang]
                    val = m.get(metric_name, 0)
                    if metric_name == 'density':
                        print(f"{val:<10.4f}", end=" ")
                    else:
                        print(f"{val:<10}", end=" ")
            print()
    
    return graphs, all_metrics


if __name__ == "__main__":
    files = {
        "en": "Bible_eng.pdf",
        "ru": "Bible_rus.pdf",
        "rom": "Bible_rom.pdf"
    }
    
    missing_files = [f for f in files.values() if not os.path.exists(f)]
    graphs, metrics = analyze_all_languages()

def integrate_extended_analysis():
    from extended_pipeline import GraphAnalysisPipeline
    files = {
    "en": "Bible_eng.pdf",
    "ru": "Bible_rus.pdf",
    "rom": "Bible_rom.pdf"
    }

    pipeline = GraphAnalysisPipeline(output_dir="extended_analysis")
    
    for lang, filename in files.items():
        if os.path.exists(filename):            
            if lang == "rom":
                text = pdf_to_text(filename, lang=lang, start_page=34)
            else:
                text = pdf_to_text(filename, lang=lang)
            
            if text:
                result = pipeline.run_full_analysis(text[:50000], lang=lang)
                
                print(f"Human graph: {result['human_metrics']['num_vertices']} vertices, "
                      f"{result['human_metrics']['num_edges']} edges")
                print(f"Bot graph: {result['bot_metrics']['num_vertices']} vertices, "
                      f"{result['bot_metrics']['num_edges']} edges")

if __name__ == "__main__":
    integrate_extended_analysis()
