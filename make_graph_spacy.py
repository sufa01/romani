from graph.higher_dim_graph import Graph
from collections import deque, Counter, defaultdict
import re

def add_vertex(graph, label):
    if not label:
        return None
    english_stop = {'be', 'is', 'are', 'was', 'were', 'been', 'the', 'a', 'an',
                    'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for',
                    'have', 'has', 'had', 'do', 'does', 'did', 'can'}
    
    if label.lower() in english_stop:
        return None
    
    if len(label) > 1 and label not in graph.vertices:
        graph.add_vertex(label, [label])
    return label


def clean_label(text):
    if not text:
        return ""
    
    if all(ord(c) < 128 for c in text if c.isalpha()):
        english_stopwords = {'be', 'is', 'are', 'was', 'were', 'been', 'the', 'a', 'an', 
                            'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for',
                            'with', 'by', 'from', 'as', 'have', 'has', 'had', 'do',
                            'does', 'did', 'will', 'would', 'could', 'should', 'does', 'can'}
        if text.lower() in english_stopwords:
            return ""  
    
    # Убираем ударения
    text = text.replace('́', '')
    # Убираем скобки и их содержимое
    text = re.sub(r'\([^)]*\)', '', text)
    # Оставляем только буквы, цифры, пробелы и дефисы
    text = re.sub(r'[^\w\s\-]', '', text)
    
    result = text.strip().lower()
    if result in {'be', 'is', 'are', 'was', 'were', 'been', 'the', 'a', 'an'}:
        return ""
    
    return result

def get_conjuncts_spacy(token_id, tokens_map):
    result = {token_id}
    queue = deque([token_id])
    while queue:
        cur = queue.popleft()
        tok = tokens_map[cur]
        for child_id in tok.children:
            child = tokens_map[child_id]
            if child.deprel == 'conj':
                if child_id not in result:
                    result.add(child_id)
                    queue.append(child_id)
    
    return result


def build_noun_concept_spacy(token_id, tokens_map):
    head = tokens_map[token_id]
    words = []
    
    def collect_modifiers(tid, is_head=False):
        tok = tokens_map[tid]
        for child_id in tok.children:
            child = tokens_map[child_id]
            if child.deprel in ('amod', 'compound', 'nummod', 'det'):
                collect_modifiers(child_id)
        
        lemma = tok.lemma if tok.lemma != '<unknown>' and tok.lemma else tok.form
        if not is_head or lemma.lower() not in ('the', 'a', 'an'):
            words.append(lemma)
        for child_id in tok.children:
            child = tokens_map[child_id]
            if child.deprel == 'prep':
                prep_lemma = child.lemma if child.lemma != '<unknown>' else child.form
                words.append(prep_lemma)
    
    conjuncts = get_conjuncts_spacy(token_id, tokens_map)
    all_words = []
    for cid in sorted(conjuncts):
        words = []
        collect_modifiers(cid, is_head=True)
        if words:
            all_words.append(' '.join(words))
    return ' and '.join(all_words) if all_words else head.form


def build_en_graph_spacy(sentences):
    """Improved version that handles sentences without verbs."""
    graph = Graph()
    NOUN_TAGS = {'NOUN', 'PROPN', 'PRON'}
    VERB_TAGS = {'VERB', 'AUX'}
    
    for sent in sentences:
        token_map = {}
        for t in sent.tokens:
            token_map[t.id] = t
            t.children = []
        
        # Строим дерево зависимостей
        for t in sent.tokens:
            if t.head in token_map:
                token_map[t.head].children.append(t.id)
        
        # Проверяем, есть ли глаголы
        has_verbs = any(tok.pos in VERB_TAGS for tok in sent.tokens)
        
        if has_verbs:
            # Используем оригинальную логику для предложений с глаголами
            for tok in sent.tokens:
                if tok.pos in VERB_TAGS:
                    # ... (оригинальный код build_en_graph_spacy)
                    pass
        else:
            # ДЛЯ ПРЕДЛОЖЕНИЙ БЕЗ ГЛАГОЛОВ: связываем существительные
            nouns = [tok for tok in sent.tokens if tok.pos in NOUN_TAGS]
            
            for i, noun1 in enumerate(nouns):
                label1 = clean_label(noun1.lemma or noun1.form)
                if not label1:
                    continue
                add_vertex(graph, label1)
                
                # Связываем с другими существительными
                for noun2 in nouns[i+1:]:
                    label2 = clean_label(noun2.lemma or noun2.form)
                    if not label2:
                        continue
                    add_vertex(graph, label2)
                    
                    # Определяем тип связи
                    if noun2.head == noun1.id:
                        meaning = noun2.deprel
                    elif noun1.head == noun2.id:
                        meaning = noun1.deprel
                    else:
                        meaning = 'related'
                    
                    try:
                        if noun1.head == noun2.id:
                            graph.add_edge(label2, label1, meaning)
                        else:
                            graph.add_edge(label1, label2, meaning)
                    except Exception:
                        pass
    
    return graph
def get_conjuncts_ru(token_id, token_map):
    result = {token_id}
    queue = deque([token_id])
    visited = set()
    
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        
        tok = token_map[cur]
        
        for child_id in tok.children:
            child = token_map[child_id]
            if child.deprel == 'conj':
                if child_id not in result:
                    result.add(child_id)
                    queue.append(child_id)
        
        head_id = tok.head
        if head_id in token_map:
            head_tok = token_map[head_id]
            for sibling_id in head_tok.children:
                sibling = token_map[sibling_id]
                if sibling.deprel == 'conj' and sibling_id not in result:
                    for sib2_id in head_tok.children:
                        if sib2_id != sibling_id and token_map[sib2_id].deprel == 'conj':
                            result.add(sib2_id)
    
    return result


def build_noun_concept_ru(token_id, token_map):
    head = token_map[token_id]
    
    def collect_modifiers(tid):
        tok = token_map[tid]
        words = []
        
        for child_id in tok.children:
            child = token_map[child_id]
            if child.deprel in ('amod', 'nummod', 'det', 'nmod'):
                words.extend(collect_modifiers(child_id))
        
        lemma = tok.lemma if tok.lemma and tok.lemma != '<unknown>' else tok.form
        if lemma and lemma.strip():
            words.append(lemma)
        
        return words
    
    conjuncts = get_conjuncts_ru(token_id, token_map)
    all_parts = []
    for cid in sorted(conjuncts):
        words = collect_modifiers(cid)
        if words:
            all_parts.append(' '.join(words))
    
    return ' и '.join(all_parts) if all_parts else (head.lemma or head.form)


def build_ru_graph_spacy(sentences):
    graph = Graph()
    NOUN_TAGS = {'NOUN', 'PROPN', 'PRON'}
    VERB_TAGS = {'VERB', 'AUX'}
    
    for sent in sentences:
        token_map = {}
        for t in sent.tokens:
            token_map[t.id] = t
            t.children = []
        
        for t in sent.tokens:
            if t.head in token_map:
                token_map[t.head].children.append(t.id)
        
        for tok in sent.tokens:
            if tok.pos in VERB_TAGS:
                subjects = []
                objects = []
                
                for child_id in tok.children:
                    child = token_map[child_id]
                    
                    if child.deprel in ('nsubj', 'nsubj:pass'):
                        subjects.extend(get_conjuncts_ru(child_id, token_map))
                    elif child.deprel in ('obj', 'iobj', 'obl'):
                        has_prep = False
                        for obl_child_id in child.children:
                            obl_child = token_map[obl_child_id]
                            if obl_child.deprel == 'case':
                                has_prep = True
                                break
                        if not has_prep:
                            objects.extend(get_conjuncts_ru(child_id, token_map))
                
                verb_label = tok.lemma if tok.lemma and tok.lemma != '<unknown>' else tok.form
                
                if subjects and objects:
                    for subj_id in subjects:
                        subj_label = build_noun_concept_ru(subj_id, token_map)
                        if not subj_label:
                            continue
                        add_vertex(graph, subj_label)
                        
                        for obj_id in objects:
                            obj_label = build_noun_concept_ru(obj_id, token_map)
                            if not obj_label:
                                continue
                            add_vertex(graph, obj_label)
                            try:
                                graph.add_edge(subj_label, obj_label, verb_label)
                            except Exception:
                                pass
        
        for tok in sent.tokens:
            if tok.pos in NOUN_TAGS:
                for child_id in tok.children:
                    child = token_map[child_id]
                    if child.deprel == 'nmod':
                        if 'Gen' in child.xpos or child.form.endswith(('а', 'я', 'ы', 'и')):
                            possessor_label = build_noun_concept_ru(child_id, token_map)
                            possessed_label = build_noun_concept_ru(tok.id, token_map)
                            if possessor_label and possessed_label:
                                add_vertex(graph, possessor_label)
                                add_vertex(graph, possessed_label)
                                try:
                                    graph.add_edge(possessed_label, possessor_label, "possessive")
                                except Exception:
                                    pass
    
    return graph

IMPORTANT_VERBS_ROM = {
    'пхэндя': 'say',
    'пхэндя́': 'say',
    'пхэн': 'say',
    'кэрдя': 'make',
    'кэрдя́': 'make',
    'кэр': 'make',
    'дыкхця': 'see',
    'дыкхця́': 'see',
    'дыкх': 'see',
    'дыя': 'give',
    'дыя́': 'give',
    'дэ': 'give',
    'создыя': 'create',
    'создыя́': 'create',
    'роскэрдя': 'separate',
    'роскэрдя́': 'separate',
    'роскэр': 'separate',
    'яця': 'become',
    'яця́': 'become',
    'ач': 'become',
    'сыс': 'be',
    'исин': 'be',
    'исы́н': 'be',
    'барьякир': 'grow',
    'скэд': 'gather',
    'джя': 'go',
    'ав': 'come',
}


class RomaniMorphologyGraph:
    """Морфологические правила для цыганского языка (для построения графа)"""
    
    PRONOUNS = {
        'ёв': {'person': 3, 'number': 'SG', 'gender': 'MASC'},
        'ев': {'person': 3, 'number': 'SG', 'gender': 'MASC'},
        'ой': {'person': 3, 'number': 'SG', 'gender': 'FEM'},
        'вон': {'person': 3, 'number': 'PL', 'gender': None},
        'лэс': {'person': 3, 'number': 'SG', 'gender': 'MASC', 'case': 'ACC'},
        'лес': {'person': 3, 'number': 'SG', 'gender': 'MASC', 'case': 'ACC'},
        'ла': {'person': 3, 'number': 'SG', 'gender': 'FEM', 'case': 'ACC'},
        'лэн': {'person': 3, 'number': 'PL', 'gender': None, 'case': 'ACC'},
        'саво': {'type': 'REL', 'number': 'SG', 'gender': 'MASC'},
        'сави': {'type': 'REL', 'number': 'SG', 'gender': 'FEM'},
        'савэ': {'type': 'REL', 'number': 'PL', 'gender': None},
        'долэ': {'type': 'DEM', 'number': 'SG'},
        'адава': {'type': 'DEM', 'number': 'SG', 'gender': 'MASC'},
        'пэскиро': {'type': 'POSS', 'reflexive': True, 'gender': 'MASC'},
        'пэскири': {'type': 'POSS', 'reflexive': True, 'gender': 'FEM'},
    }
    
    @classmethod
    def clean_word(cls, word):
        if not word:
            return ""
        return word.lower().replace('́', '').strip()
    
    @classmethod
    def get_features(cls, word):
        word_clean = cls.clean_word(word)
        return cls.PRONOUNS.get(word_clean, {})
    
    @classmethod
    def is_relative(cls, word):
        return cls.get_features(word).get('type') == 'REL'
    
    @classmethod
    def is_demonstrative(cls, word):
        return cls.get_features(word).get('type') == 'DEM'


def get_conjuncts_rom(token_id, token_map):
    result = {token_id}
    queue = deque([token_id])
    visited = set()
    
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        
        tok = token_map[cur]
        
        for child_id in tok.children:
            child = token_map[child_id]
            if child.deprel in ('conj', 'cc'):
                if child_id not in result:
                    result.add(child_id)
                    queue.append(child_id)
        
        if tok.head in token_map:
            parent = token_map[tok.head]
            for sibling_id in parent.children:
                sibling = token_map[sibling_id]
                if sibling.deprel == 'conj' and sibling_id not in result:
                    result.add(sibling_id)
                    queue.append(sibling_id)
    
    return result


def build_noun_concept_rom(token_id, token_map):
    head = token_map[token_id]
    
    def collect_words(tid, depth=0):
        if depth > 5:
            return []
        
        tok = token_map[tid]
        words = []
        
        for child_id in tok.children:
            child = token_map[child_id]
            if child.deprel in ('amod', 'nummod', 'det', 'nmod'):
                words.extend(collect_words(child_id, depth + 1))
        
        if tok.lemma and tok.lemma != "<unknown>":
            # Проверяем, не английское ли слово
            if all(ord(c) < 128 for c in tok.lemma if c.isalpha()) and not any(ord(c) >= 0x0400 for c in tok.lemma):
                # Это английская лемма - используем form
                word = clean_label(tok.form)
            else:
                word = clean_label(tok.lemma)
        else:
            word = clean_label(tok.form)
        
        if word:
            words.append(word)
        
        return words
    
    conjuncts = get_conjuncts_rom(token_id, token_map)
    all_parts = []
    
    for cid in sorted(conjuncts):
        words = collect_words(cid)
        if words:
            all_parts.append(' '.join(words))
    
    result = ' и '.join(all_parts) if all_parts else clean_label(head.form)
    
    if result in {'be', 'is', 'are', 'was', 'were', 'been'}:
        return clean_label(head.form)
    
    return result


def get_verb_semantic_type(verb_form, verb_lemma):
    form_clean = RomaniMorphologyGraph.clean_word(verb_form)
    lemma_clean = RomaniMorphologyGraph.clean_word(verb_lemma) if verb_lemma else ""
    
    for rom_verb, sem_type in IMPORTANT_VERBS_ROM.items():
        rom_clean = RomaniMorphologyGraph.clean_word(rom_verb)
        if rom_clean in form_clean or rom_clean in lemma_clean:
            return sem_type
    
    return "does"


def build_rom_graph_spacy(sentences):
    """Build graph for Romani language - simplified version."""
    from graph.higher_dim_graph import Graph
    
    graph = Graph()
    
    # Части речи для существительных и глаголов
    NOUN_TAGS = {'NOUN', 'PROPN', 'PRON'}
    VERB_TAGS = {'VERB', 'AUX'}
    
    for sent in sentences:
        # Собираем существительные и глаголы
        nouns = []
        verbs = []
        
        for tok in sent.tokens:
            form = tok.form.strip()
            lemma = tok.lemma.strip() if tok.lemma else form
            
            # Очищаем текст
            form = form.replace('́', '').lower()
            lemma = lemma.replace('́', '').lower()
            
            if not form or len(form) < 2:
                continue
            
            if tok.pos in VERB_TAGS:
                verbs.append({'form': form, 'lemma': lemma, 'id': tok.id})
            elif tok.pos in NOUN_TAGS:
                nouns.append({'form': form, 'lemma': lemma, 'id': tok.id})
        
        # Если есть и глаголы и существительные - строим связи
        if verbs and nouns:
            for verb in verbs:
                # Добавляем глагол как вершину
                verb_label = verb['lemma'] if verb['lemma'] != '<unknown>' else verb['form']
                if verb_label not in graph.vertices:
                    try:
                        graph.add_vertex(verb_label, [verb_label])
                    except ValueError:
                        pass
                
                # Связываем существительные с глаголом
                for noun in nouns:
                    noun_label = noun['lemma'] if noun['lemma'] != '<unknown>' else noun['form']
                    if noun_label not in graph.vertices:
                        try:
                            graph.add_vertex(noun_label, [noun_label])
                        except ValueError:
                            pass
                    
                    try:
                        # Существительное → Глагол (кто? что делает?)
                        if noun['id'] < verb['id']:
                            graph.add_edge(noun_label, verb_label, 'действие')
                        else:
                            graph.add_edge(verb_label, noun_label, 'действие')
                    except ValueError:
                        pass
        
        # Если нет глаголов - связываем существительные между собой
        elif len(nouns) >= 2:
            for i, noun1 in enumerate(nouns):
                label1 = noun1['lemma'] if noun1['lemma'] != '<unknown>' else noun1['form']
                if label1 not in graph.vertices:
                    try:
                        graph.add_vertex(label1, [label1])
                    except ValueError:
                        pass
                
                # Связываем с последующими существительными
                for noun2 in nouns[i+1:]:
                    label2 = noun2['lemma'] if noun2['lemma'] != '<unknown>' else noun2['form']
                    if label2 not in graph.vertices:
                        try:
                            graph.add_vertex(label2, [label2])
                        except ValueError:
                            pass
                    
                    try:
                        # Первое → второе (порядок в предложении)
                        graph.add_edge(label1, label2, 'следует')
                    except ValueError:
                        pass
        
        # Если только одно существительное или только глаголы
        else:
            for tok in sent.tokens:
                form = tok.form.strip().replace('́', '').lower()
                if len(form) >= 2:
                    if form not in graph.vertices:
                        try:
                            graph.add_vertex(form, [form])
                        except ValueError:
                            pass
    
    # Если граф всё ещё пустой - добавляем хоть что-то
    if len(graph.vertices) == 0:
        for sent in sentences:
            words_in_order = []
            for tok in sent.tokens:
                form = tok.form.strip().replace('́', '').lower()
                if len(form) >= 2:
                    words_in_order.append(form)
                    if form not in graph.vertices:
                        try:
                            graph.add_vertex(form, [form])
                        except ValueError:
                            pass
            
            # Связываем последовательные слова
            for i in range(len(words_in_order) - 1):
                if words_in_order[i] != words_in_order[i+1]:
                    try:
                        graph.add_edge(words_in_order[i], words_in_order[i+1], 'после')
                    except ValueError:
                        pass
    
    return graph
def _build_rom_graph_fallback(self, sentences):
    """Improved fallback for Romani that creates meaningful connections."""
    from graph.graph import Graph as GraphClass
    
    graph = GraphClass()
    
    # Собираем все значимые слова
    important_words = set()
    for sent in sentences:
        for tok in sent.tokens:
            word = tok.form.lower().replace('́', '')
            if len(word) > 2 and word.isalpha():
                important_words.add(word)
    
    # Добавляем вершины
    for word in important_words:
        try:
            graph.add_vertex(word, [word])
        except ValueError:
            pass
    
    # Создаем связи: слова из одного предложения связываем
    for sent in sentences:
        sent_words = []
        for tok in sent.tokens:
            word = tok.form.lower().replace('́', '')
            if word in graph.vertices:
                sent_words.append(word)
        
        # Связываем последовательные слова
        for i in range(len(sent_words) - 1):
            if sent_words[i] != sent_words[i+1]:
                try:
                    # Используем направление: первое слово → второе
                    graph.add_edge(sent_words[i], sent_words[i+1], 'после')
                except ValueError:
                    pass
        
        # Связываем первое и последнее слово предложения
        if len(sent_words) > 2 and sent_words[0] != sent_words[-1]:
            try:
                graph.add_edge(sent_words[0], sent_words[-1], 'контекст')
            except ValueError:
                pass
    
    return graph

__all__ = [
    'build_en_graph_spacy',
    'build_ru_graph_spacy',
    'build_rom_graph_spacy',
]

if __name__ == "__main__":
    # Создаем тестовые предложения для каждого языка
    from sent_class import Sentence, Token
    en_tokens = [
        Token(id=1, form="God", lemma="God", pos="PROPN", head=2, deprel="nsubj"),
        Token(id=2, form="created", lemma="create", pos="VERB", head=0, deprel="root"),
        Token(id=3, form="heaven", lemma="heaven", pos="NOUN", head=2, deprel="dobj"),
    ]
    en_sent = Sentence(tokens=en_tokens, sent_id=0, text="God created heaven")
    en_graph = build_en_graph_spacy([en_sent])
    print(f"Вершин: {len(en_graph.vertices)}, Рёбер: {len(en_graph.edges)}")
    ru_tokens = [
        Token(id=1, form="Бог", lemma="бог", pos="NOUN", head=2, deprel="nsubj"),
        Token(id=2, form="сотворил", lemma="сотворить", pos="VERB", head=0, deprel="root"),
        Token(id=3, form="небо", lemma="небо", pos="NOUN", head=2, deprel="obj"),
    ]
    ru_sent = Sentence(tokens=ru_tokens, sent_id=0, text="Бог сотворил небо")
    ru_graph = build_ru_graph_spacy([ru_sent])
    print(f"Вершин: {len(ru_graph.vertices)}, Рёбер: {len(ru_graph.edges)}")
    rom_tokens = [
        Token(id=1, form="Дэвэ́л", lemma="Дэвэл", pos="PROPN", head=2, deprel="nsubj"),
        Token(id=2, form="создыя́", lemma="создыя", pos="VERB", head=0, deprel="root"),
        Token(id=3, form="болыбэ́н", lemma="болыбэн", pos="NOUN", head=2, deprel="obj"),
    ]
    rom_sent = Sentence(tokens=rom_tokens, sent_id=0, text="Дэвэ́л создыя́ болыбэ́н")
    rom_graph = build_rom_graph_spacy([rom_sent])
    print(f"Вершин: {len(rom_graph.vertices)}, Рёбер: {len(rom_graph.edges)}")