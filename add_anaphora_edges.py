import re


def clean_label_simple(text):
    """очистка метки без фильтрации языка"""
    if not text:
        return ""
    text = text.replace('́', '')
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[^\w\s\-]', '', text)
    return text.strip().lower()


def add_anaphora_edges_en(graph, sentences):
    """Добавляет рёбра анафоры для английского графа"""
    edges_added = 0
    
    for sent in sentences:
        for tok in sent.tokens:
            if tok.pos == "PRON":
                misc = getattr(tok, 'misc', '')
                if misc and 'Antecedent=' in misc:
                    antecedent = misc.split('Antecedent=')[1].strip()
                    
                    pronoun_clean = clean_label_simple(tok.form)
                    antecedent_clean = clean_label_simple(antecedent)
                    
                    if (pronoun_clean and antecedent_clean and 
                        pronoun_clean in graph.vertices and 
                        antecedent_clean in graph.vertices and
                        pronoun_clean != antecedent_clean):
                        try:
                            graph.add_edge(pronoun_clean, antecedent_clean, "refers_to")
                            edges_added += 1
                        except:
                            pass
    
    return edges_added


def add_anaphora_edges_ru(graph, sentences):
    """Добавляет рёбра анафоры для русского графа"""
    edges_added = 0
    
    for sent in sentences:
        for tok in sent.tokens:
            if tok.pos == "PRON" or tok.pos.startswith("P-"):
                misc = getattr(tok, 'misc', '')
                if misc and 'Antecedent=' in misc:
                    antecedent = misc.split('Antecedent=')[1].strip()
                    
                    pronoun_clean = clean_label_simple(tok.form)
                    antecedent_clean = clean_label_simple(antecedent)
                    
                    if (pronoun_clean and antecedent_clean and 
                        pronoun_clean in graph.vertices and 
                        antecedent_clean in graph.vertices and
                        pronoun_clean != antecedent_clean):
                        try:
                            graph.add_edge(pronoun_clean, antecedent_clean, "указывает_на")
                            edges_added += 1
                        except:
                            pass
    
    return edges_added


def add_anaphora_edges_rom(graph, sentences):
    """Добавляет рёбра анафоры для цыганского графа"""
    edges_added = 0
    
    for sent in sentences:
        for tok in sent.tokens:
            if tok.pos == "PRON":
                misc = getattr(tok, 'misc', '')
                if misc and 'Antecedent=' in misc:
                    antecedent = misc.split('Antecedent=')[1].strip()
                    
                    pronoun_clean = clean_label_simple(tok.form)
                    antecedent_clean = clean_label_simple(antecedent)
                    
                    if (pronoun_clean and antecedent_clean and 
                        pronoun_clean in graph.vertices and 
                        antecedent_clean in graph.vertices and
                        pronoun_clean != antecedent_clean):
                        try:
                            graph.add_edge(pronoun_clean, antecedent_clean, "сыкавэл_пэ")  # "указывает_на" по-цыгански
                            edges_added += 1
                        except:
                            pass
    
    return edges_added


def add_anaphora_edges(graph, sentences, lang):
    """Универсальная функция добавления рёбер анафоры"""
    if lang == "en":
        return add_anaphora_edges_en(graph, sentences)
    elif lang == "ru":
        return add_anaphora_edges_ru(graph, sentences)
    elif lang == "rom":
        return add_anaphora_edges_rom(graph, sentences)
    else:
        return 0
