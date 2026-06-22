# romani_rb_anaphora.py
"""
Разрешение анафоры для цыганского языка (диалект из текста Бытия)
Основано на rule-based подходе с учетом морфологии и синтаксиса цыганского
"""

from sent_class import parse_conll
from collections import defaultdict, deque
import re


class RomaniMorphology:
    """
    Морфологические правила для цыганского языка (диалект из текста)
    """
    
    # Словарь местоимений с их свойствами
    PRONOUNS = {
        # Личные местоимения (именительный падеж)
        "мэ": {"person": 1, "number": "SG", "case": "NOM", "gender": None, "gloss": "я"},
        "ме": {"person": 1, "number": "SG", "case": "NOM", "gender": None, "gloss": "я"},
        "ту": {"person": 2, "number": "SG", "case": "NOM", "gender": None, "gloss": "ты"},
        "ёв": {"person": 3, "number": "SG", "case": "NOM", "gender": "MASC", "gloss": "он"},
        "ев": {"person": 3, "number": "SG", "case": "NOM", "gender": "MASC", "gloss": "он"},
        "йов": {"person": 3, "number": "SG", "case": "NOM", "gender": "MASC", "gloss": "он"},
        "ой": {"person": 3, "number": "SG", "case": "NOM", "gender": "FEM", "gloss": "она"},
        "амен": {"person": 1, "number": "PL", "case": "NOM", "gender": None, "gloss": "мы"},
        "амэн": {"person": 1, "number": "PL", "case": "NOM", "gender": None, "gloss": "мы"},
        "тумен": {"person": 2, "number": "PL", "case": "NOM", "gender": None, "gloss": "вы"},
        "тумэн": {"person": 2, "number": "PL", "case": "NOM", "gender": None, "gloss": "вы"},
        "вон": {"person": 3, "number": "PL", "case": "NOM", "gender": None, "gloss": "они"},
        
        # Косвенные падежи (аккузатив/датив)
        "ман": {"person": 1, "number": "SG", "case": "ACC", "gender": None, "gloss": "меня"},
        "тут": {"person": 2, "number": "SG", "case": "ACC", "gender": None, "gloss": "тебя"},
        "лэс": {"person": 3, "number": "SG", "case": "ACC", "gender": "MASC", "gloss": "его"},
        "лес": {"person": 3, "number": "SG", "case": "ACC", "gender": "MASC", "gloss": "его"},
        "ла": {"person": 3, "number": "SG", "case": "ACC", "gender": "FEM", "gloss": "её"},
        "амэн": {"person": 1, "number": "PL", "case": "ACC", "gender": None, "gloss": "нас"},
        "тумэн": {"person": 2, "number": "PL", "case": "ACC", "gender": None, "gloss": "вас"},
        "лэн": {"person": 3, "number": "PL", "case": "ACC", "gender": None, "gloss": "их"},
        "лен": {"person": 3, "number": "PL", "case": "ACC", "gender": None, "gloss": "их"},
        
        # Датив
        "мангэ": {"person": 1, "number": "SG", "case": "DAT", "gender": None, "gloss": "мне"},
        "тукэ": {"person": 2, "number": "SG", "case": "DAT", "gender": None, "gloss": "тебе"},
        "лэскэ": {"person": 3, "number": "SG", "case": "DAT", "gender": "MASC", "gloss": "ему"},
        "лакэ": {"person": 3, "number": "SG", "case": "DAT", "gender": "FEM", "gloss": "ей"},
        
        # Притяжательные местоимения
        "миро": {"person": 1, "number": "SG", "gender": "MASC", "type": "POSS", "gloss": "мой"},
        "мири": {"person": 1, "number": "SG", "gender": "FEM", "type": "POSS", "gloss": "моя"},
        "мирэ": {"person": 1, "number": "PL", "type": "POSS", "gloss": "мои"},
        "тиро": {"person": 2, "number": "SG", "gender": "MASC", "type": "POSS", "gloss": "твой"},
        "тири": {"person": 2, "number": "SG", "gender": "FEM", "type": "POSS", "gloss": "твоя"},
        "лэско": {"person": 3, "number": "SG", "gender": "MASC", "type": "POSS", "gloss": "его"},
        "лэски": {"person": 3, "number": "SG", "gender": "FEM", "type": "POSS", "gloss": "его"},
        "лако": {"person": 3, "number": "SG", "gender": "MASC", "type": "POSS", "gloss": "её"},
        "лаки": {"person": 3, "number": "SG", "gender": "FEM", "type": "POSS", "gloss": "её"},
        "амаро": {"person": 1, "number": "PL", "gender": "MASC", "type": "POSS", "gloss": "наш"},
        "амари": {"person": 1, "number": "PL", "gender": "FEM", "type": "POSS", "gloss": "наша"},
        "тумаро": {"person": 2, "number": "PL", "gender": "MASC", "type": "POSS", "gloss": "ваш"},
        "тумари": {"person": 2, "number": "PL", "gender": "FEM", "type": "POSS", "gloss": "ваша"},
        "лэнго": {"person": 3, "number": "PL", "gender": "MASC", "type": "POSS", "gloss": "их"},
        "лэнги": {"person": 3, "number": "PL", "gender": "FEM", "type": "POSS", "gloss": "их"},
        
        # Возвратно-притяжательное
        "пэскиро": {"person": 3, "reflexive": True, "type": "POSS", "gender": "MASC", "gloss": "свой"},
        "пэскири": {"person": 3, "reflexive": True, "type": "POSS", "gender": "FEM", "gloss": "своя"},
        "пэско": {"person": 3, "reflexive": True, "type": "POSS", "gender": "MASC", "gloss": "свой"},
        "пэскирэ": {"person": 3, "reflexive": True, "type": "POSS", "number": "PL", "gloss": "свои"},
        
        # Указательные местоимения
        "адава": {"type": "DEM", "number": "SG", "gender": "MASC", "gloss": "этот"},
        "адая": {"type": "DEM", "number": "SG", "gender": "FEM", "gloss": "эта"},
        "адалэ": {"type": "DEM", "number": "PL", "gloss": "эти"},
        "дава": {"type": "DEM", "number": "SG", "gender": "MASC", "gloss": "это"},
        "долэ": {"type": "DEM", "number": "SG", "case": "OBL", "gloss": "того"},
        "дола": {"type": "DEM", "number": "SG", "case": "OBL", "gender": "FEM", "gloss": "ту"},
        "кадя": {"type": "DEM", "number": "SG", "gender": "FEM", "gloss": "эта"},
        "када": {"type": "DEM", "number": "SG", "gender": "MASC", "gloss": "этот"},
        
        # Относительные местоимения
        "саво": {"type": "REL", "number": "SG", "gender": "MASC", "gloss": "который"},
        "сави": {"type": "REL", "number": "SG", "gender": "FEM", "gloss": "которая"},
        "савэ": {"type": "REL", "number": "PL", "gloss": "которые"},
        "со": {"type": "REL", "gloss": "что/который"},
        
        # Вопросительные
        "ко": {"type": "INT", "gloss": "кто"},
        "со": {"type": "INT", "gloss": "что"},
        "саво": {"type": "INT", "gloss": "какой"},
        "кон": {"type": "INT", "gloss": "кто"},
        "кай": {"type": "INT", "gloss": "где/куда"},
    }
    
    # Маркеры рода существительных (по окончаниям)
    MASC_ENDINGS = {
        "о", "os", "es", "as", "is", "us",
        "ибэн", "ыбэн", "ипэн", "ыпэн",
        "мо", "ло", "ко", "ро", "до", "то",
        "л", "н", "р", "й",
    }
    
    FEM_ENDINGS = {
        "и", "a", "e", "ja", "ya",
        "ибны", "ыбны",
        "ата", "ита", "ица",
        "лы", "ны",
    }
    
    # Маркеры множественного числа
    PL_ENDINGS = {
        "э", "а", "я",
        "ура", "ора", "ара",
        "ибна", "ыбна", "ипна", "ыпна",
        "эн", "ен",
    }
    
    @classmethod
    def clean_word(cls, word):
        """Очищает слово от диакритики"""
        if not word:
            return ""
        return word.lower().replace('́', '').strip()
    
    @classmethod
    def get_gender(cls, word):
        """Определяет род существительного по окончанию"""
        word_clean = cls.clean_word(word)
        
        # Проверяем мужские окончания
        for ending in cls.MASC_ENDINGS:
            if word_clean.endswith(ending):
                return "MASC"
        
        # Проверяем женские окончания
        for ending in cls.FEM_ENDINGS:
            if word_clean.endswith(ending):
                return "FEM"
        
        # Эвристика: если заканчивается на согласный - мужской род
        if word_clean and word_clean[-1] not in 'аэиоуяюеёы':
            return "MASC"
        
        return "FEM"  # по умолчанию
    
    @classmethod
    def get_number(cls, word):
        """Определяет число существительного"""
        word_clean = cls.clean_word(word)
        
        for ending in cls.PL_ENDINGS:
            if word_clean.endswith(ending) and len(word_clean) > 3:
                # Дополнительная проверка: не является ли это просто окончанием
                if not word_clean.endswith(('ибэн', 'ыбэн', 'ипэн')):
                    return "PL"
        
        return "SG"
    
    @classmethod
    def get_pronoun_features(cls, word):
        """Возвращает признаки местоимения"""
        if not word:
            return {}
        word_clean = cls.clean_word(word)
        return cls.PRONOUNS.get(word_clean, {})
    
    @classmethod
    def is_third_person_pronoun(cls, word):
        """Проверяет, является ли слово местоимением 3 лица"""
        feats = cls.get_pronoun_features(word)
        return feats.get("person") == 3 or feats.get("type") in ("DEM", "REL")
    
    @classmethod
    def is_reflexive_possessive(cls, word):
        """Проверяет, является ли слово возвратно-притяжательным (пэскиро)"""
        feats = cls.get_pronoun_features(word)
        return feats.get("reflexive", False)
    
    @classmethod
    def is_relative_pronoun(cls, word):
        """Проверяет, является ли слово относительным местоимением"""
        feats = cls.get_pronoun_features(word)
        return feats.get("type") == "REL"
    
    @classmethod
    def is_demonstrative(cls, word):
        """Проверяет, является ли слово указательным местоимением"""
        feats = cls.get_pronoun_features(word)
        return feats.get("type") == "DEM"


class NPExtractor:
    """
    Извлечение именных групп для цыганского языка
    """
    
    def extract_nps(self, sent):
        nps = []
        
        for tok in sent.tokens:
            # Существительные, имена собственные и местоимения
            if tok.pos in ("NOUN", "PROPN", "PRON"):
                np_tokens = self._collect_np_tokens(tok, sent)
                
                # Определяем род и число
                gender = RomaniMorphology.get_gender(tok.form)
                number = RomaniMorphology.get_number(tok.form)
                
                # Для местоимений берем признаки из словаря
                if tok.pos == "PRON":
                    feats = RomaniMorphology.get_pronoun_features(tok.form)
                    if feats:
                        gender = feats.get("gender", gender)
                        number = feats.get("number", number)
                
                np = {
                    "head_id": tok.id,
                    "tokens": sorted(np_tokens),
                    "head_lemma": self._get_lemma(tok),
                    "head_form": tok.form,
                    "pos": tok.pos,
                    "number": number,
                    "gender": gender,
                    "person": 3,
                    "gram_role": self._get_grammatical_role(tok, sent),
                    "sentence_id": sent.sent_id
                }
                
                nps.append(np)
        
        return nps
    def _is_english(self, word):
        """Проверяет, является ли слово английским"""
        if not word:
            return False
        # Если все буквы латинские и нет кириллицы - вероятно английское
        has_cyrillic = any(ord(c) >= 0x0400 for c in word)
        return not has_cyrillic and all(ord(c) < 128 for c in word if c.isalpha())
    
    def _get_lemma(self, tok):
        """Получает лемму токена - всегда используем form если lemma неизвестна"""
        if tok.lemma and tok.lemma != "<unknown>" and not self._is_english(tok.lemma):
            return RomaniMorphology.clean_word(tok.lemma)
        return RomaniMorphology.clean_word(tok.form)
    
    def _collect_np_tokens(self, head, sent):
        """Собирает все токены, относящиеся к данной NP"""
        tokens = {head.id}
        
        for tok in sent.tokens:
            if tok.head == head.id:
                # Модификаторы, которые могут входить в NP
                if tok.deprel in ("det", "amod", "nmod", "nummod", "case", "compound"):
                    tokens.add(tok.id)
                # Прилагательные
                elif tok.pos == "ADJ":
                    tokens.add(tok.id)
                # Притяжательные местоимения
                elif tok.pos == "PRON" and RomaniMorphology.get_pronoun_features(tok.form).get("type") == "POSS":
                    tokens.add(tok.id)
        
        return tokens
    
    def _get_grammatical_role(self, tok, sent):
        """Определяет грамматическую роль NP"""
        # Если это субъект
        if tok.deprel in ("nsubj", "nsubj:pass", "csubj"):
            return "SUBJ"
        
        # Если это объект
        if tok.deprel in ("obj", "iobj"):
            return "OBJ"
        
        # Косвенный объект
        if tok.deprel == "obl":
            # Проверяем, есть ли предлог
            for parent_tok in sent.tokens:
                if parent_tok.head == tok.head and parent_tok.deprel == "case":
                    return "POBJ"
            return "OBJ"
        
        # Проверяем позицию в предложении (эвристика для субъекта)
        sent_tokens = list(sent.tokens)
        try:
            tok_idx = sent_tokens.index(tok)
            # Если это первая NP в предложении и перед ней нет других NP
            if tok_idx < 5:
                for i in range(tok_idx):
                    if sent_tokens[i].pos in ("NOUN", "PROPN", "PRON"):
                        break
                else:
                    return "SUBJ"
        except ValueError:
            pass
        
        return "OTHER"


class BindingFilter:
    """
    Фильтры теории связывания для цыганского
    """
    
    def filter_coargument(self, pronoun_token, candidate_nps, sent):
        """Исключает NP, которые являются ко-аргументами того же предиката"""
        filtered = []
        
        pron_head = sent.get_token(pronoun_token.head)
        if not pron_head:
            return candidate_nps
        
        for np in candidate_nps:
            # NP из предыдущих предложений не фильтруем
            if np["sentence_id"] != sent.sent_id:
                filtered.append(np)
                continue
            
            cand_head = sent.get_token(np["head_id"])
            if not cand_head:
                filtered.append(np)
                continue
            
            # Проверяем, связаны ли с одним предикатом
            if cand_head.head == pron_head.id and pronoun_token.head == pron_head.id:
                # Но если это разные аргументы, то исключаем
                if cand_head.deprel != pronoun_token.deprel:
                    continue
            
            filtered.append(np)
        
        return filtered
    
    def filter_cataphora(self, pronoun_token, candidate_nps, sent):
        # Для относительных местоимений катафора возможна
        if RomaniMorphology.is_relative_pronoun(pronoun_token.form):
            return candidate_nps
        
        pronoun_position = pronoun_token.id
        
        filtered = []
        for np in candidate_nps:
            if np["sentence_id"] == sent.sent_id:
                # Антецедент должен быть перед местоимением
                if np["head_id"] < pronoun_position:
                    filtered.append(np)
            else:
                # Из предыдущих предложений - всегда ок
                filtered.append(np)
        
        return filtered if filtered else candidate_nps


class DiscourseModel:    
    def __init__(self, max_sent_distance=5):
        self.max_sent_distance = max_sent_distance
        self.entities = []
        self.topic_stack = []  # Стек топиков
    
    def add_nps(self, nps, current_sent_id):
        for np in nps:
            # Проверяем, нет ли уже такой сущности
            existing = self._find_existing(np)
            
            if existing:
                existing["last_seen"] = current_sent_id
                existing["mention_count"] = existing.get("mention_count", 1) + 1
                existing["salience"] += 50  # Бонус за повторное упоминание
            else:
                entry = {
                    "np": np,
                    "salience": 100 if np["gram_role"] == "SUBJ" else 50,
                    "last_seen": current_sent_id,
                    "mention_count": 1
                }
                
                # Субъекты получают приоритет в стеке топиков
                if np["gram_role"] == "SUBJ":
                    self.topic_stack.insert(0, entry)
                    if len(self.topic_stack) > 3:
                        self.topic_stack.pop()
                
                self.entities.append(entry)
    
    def _find_existing(self, np):
        for ent in self.entities:
            if ent["np"]["head_lemma"] == np["head_lemma"]:
                return ent
        return None
    
    def decay(self, current_sent_id):
        new_entities = []
        
        for ent in self.entities:
            distance = current_sent_id - ent["last_seen"]
            
            if distance > self.max_sent_distance:
                continue
            
            # Экспоненциальное затухание салиентности
            ent["salience"] = ent["salience"] / (2 ** distance)
            new_entities.append(ent)
        
        self.entities = new_entities
        
        # Также обновляем стек топиков
        self.topic_stack = [e for e in self.topic_stack 
                           if current_sent_id - e["last_seen"] <= self.max_sent_distance]
    
    def get_candidates(self, current_sent_id):
        candidates = []
        
        # Сначала добавляем топики (они более салиентны)
        for ent in self.topic_stack:
            if current_sent_id - ent["last_seen"] <= self.max_sent_distance:
                candidates.append(ent["np"])
        
        # Затем остальные сущности
        for ent in self.entities:
            if current_sent_id - ent["last_seen"] <= self.max_sent_distance:
                if ent["np"] not in candidates:
                    candidates.append(ent["np"])
        
        return candidates


class SalienceScorer:
    ROLE_WEIGHTS = {
        "SUBJ": 100,   # Субъекты в цыганском - главные кандидаты
        "OBJ": 60,
        "POBJ": 40,
        "OTHER": 20
    }
    
    RECENCY_WEIGHT = 120  # Недавность очень важна
    PARALLELISM_BONUS = 40  # Параллелизм конструкций
    GENDER_MATCH_BONUS = 50
    NUMBER_MATCH_BONUS = 40
    MENTION_COUNT_BONUS = 20  # Бонус за частоту упоминания
    
    def score(self, pronoun_token, candidates, sent):
        scored = []
        
        pron_features = RomaniMorphology.get_pronoun_features(pronoun_token.form)
        
        for np in candidates:
            weight = 0
            
            # расстояние в предложениях)
            distance = sent.sent_id - np["sentence_id"]
            weight += self.RECENCY_WEIGHT / (distance + 1)
            
            # Грамматическая роль
            weight += self.ROLE_WEIGHTS.get(np["gram_role"], 0)
            
            # Согласование по роду и числу
            if pron_features:
                if np.get("gender") == pron_features.get("gender"):
                    weight += self.GENDER_MATCH_BONUS
                if np.get("number") == pron_features.get("number"):
                    weight += self.NUMBER_MATCH_BONUS
            else:
                # Если признаки местоимения не найдены, определяем эвристически
                pron_form = RomaniMorphology.clean_word(pronoun_token.form)
                
                # Для 'ёв'/'ев' - мужской род ед.ч.
                if pron_form in ('ёв', 'ев', 'йов', 'лэс', 'лес'):
                    if np.get("gender") == "MASC" and np.get("number") == "SG":
                        weight += 80
                
                # Для 'ой' - женский род ед.ч.
                elif pron_form in ('ой', 'ла'):
                    if np.get("gender") == "FEM" and np.get("number") == "SG":
                        weight += 80
                
                # Для 'вон'/'лэн' - мн.ч.
                elif pron_form in ('вон', 'лэн', 'лен'):
                    if np.get("number") == "PL":
                        weight += 80
            
            # Параллелизм: если местоимение и NP в одной роли
            pron_role = self._get_pronoun_role(pronoun_token)
            if pron_role == np["gram_role"]:
                weight += self.PARALLELISM_BONUS
            # Дэвэл (Бог) - всегда высокоприоритетный кандидат
            if 'дэвэл' in np.get("head_lemma", "").lower():
                weight += 100
            
            scored.append((np, weight))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _get_pronoun_role(self, pron_token):
        if pron_token.deprel in ("nsubj", "nsubj:pass"):
            return "SUBJ"
        if pron_token.deprel in ("obj", "iobj"):
            return "OBJ"
        return "OTHER"


class MorphFilter:
    def filter(self, pronoun_token, candidates):
        pron_form = RomaniMorphology.clean_word(pronoun_token.form)
        pron_features = RomaniMorphology.get_pronoun_features(pron_form)
        
        # Если это не местоимение 3 лица или относительное/указательное
        if not pron_features:
            # Проверяем известные формы 3 лица
            if pron_form in ('ёв', 'ев', 'йов', 'ой', 'вон', 'лэс', 'лес', 'ла', 'лэн', 'лен',
                           'саво', 'сави', 'савэ', 'долэ', 'адава'):
                pron_features = {
                    "person": 3,
                    "number": "PL" if pron_form in ('вон', 'лэн', 'лен', 'савэ') else "SG",
                    "gender": "FEM" if pron_form in ('ой', 'ла', 'сави') else "MASC"
                }
            else:
                # Для неизвестных местоимений возвращаем всех кандидатов
                return candidates
        
        filtered = []
        for np in candidates:
            # Пропускаем местоимения 1 и 2 лица в качестве антецедентов
            np_form = RomaniMorphology.clean_word(np["head_form"])
            np_feats = RomaniMorphology.get_pronoun_features(np_form)
            if np_feats and np_feats.get("person") in (1, 2):
                continue
            
            # Проверяем число
            if pron_features.get("number") and np.get("number"):
                if pron_features["number"] != np["number"]:
                    # Исключение: собирательные существительные
                    if not (np_form.endswith(('ибэн', 'ыбэн')) and pron_features["number"] == "SG"):
                        continue
            
            # Проверяем род (только для единственного числа)
            if pron_features.get("number") == "SG" and pron_features.get("gender"):
                if np.get("gender") and pron_features["gender"] != np["gender"]:
                    # Исключение: слова среднего рода или без рода
                    if np.get("gender") not in ("MASC", "FEM"):
                        pass
                    else:
                        continue
            
            filtered.append(np)
        
        return filtered if filtered else candidates


def resolve_anaphora_rom(conll_file: str):
    sentences = parse_conll(conll_file)
    
    ENGLISH_STOP = {'be', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
                    'do', 'does', 'did', 'the', 'a', 'an', 'and', 'or', 'but', 'of',
                    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'can'}
    
    for sent in sentences:
        filtered_tokens = []
        for tok in sent.tokens:
            if tok.form.lower() in ENGLISH_STOP or tok.lemma.lower() in ENGLISH_STOP:
                continue
            if all(ord(c) < 128 for c in tok.form if c.isalpha()):
                continue
            filtered_tokens.append(tok)
        
        if filtered_tokens:
            for i, tok in enumerate(filtered_tokens, 1):
                tok.id = i
                if tok.head > len(filtered_tokens):
                    tok.head = 0
            sent.tokens = filtered_tokens
            sent.text = ' '.join(t.form for t in filtered_tokens)
    
    np_extractor = NPExtractor()
    binding = BindingFilter()
    dm = DiscourseModel(max_sent_distance=5)
    scorer = SalienceScorer()
    morph_filter = MorphFilter()
    
    resolved_count = 0
    antecedents_found = []
    
    total_nps = 0
    total_pronouns_found = 0
    total_candidates = 0
    
    for sent_idx, sent in enumerate(sentences):
        # Извлекаем NP из текущего предложения
        nps = np_extractor.extract_nps(sent)
        total_nps += len(nps)
        
        # Обновляем дискурсную модель
        dm.decay(sent.sent_id)
        
        # Находим все местоимения для разрешения
        pronouns = []
        for t in sent.tokens:
            form_clean = RomaniMorphology.clean_word(t.form)
            
            if sent_idx < 5 and (t.pos == "PRON" or form_clean in RomaniMorphology.PRONOUNS):
                print(f"  [DEBUG] Предл {sent_idx}: найдено местоимение '{t.form}' (pos={t.pos})")
            
            if t.pos == "PRON" or form_clean in RomaniMorphology.PRONOUNS:
                feats = RomaniMorphology.get_pronoun_features(form_clean)
                
                if feats:
                    if (feats.get("person") == 3 or 
                        feats.get("type") in ("REL", "DEM")):
                        pronouns.append(t)
                        total_pronouns_found += 1
                elif form_clean in ('ёв', 'ев', 'йов', 'ой', 'вон', 'лэс', 'лес', 'ла', 'лэн', 'лен',
                                   'саво', 'сави', 'савэ', 'долэ', 'адава', 'дава'):
                    pronouns.append(t)
                    total_pronouns_found += 1
        
        for pron in pronouns:
            candidates = dm.get_candidates(sent.sent_id)
            total_candidates += len(candidates)
            
            if not candidates:
                continue
            
            filtered = binding.filter_coargument(pron, candidates, sent)
            filtered = binding.filter_cataphora(pron, filtered, sent)
            filtered = morph_filter.filter(pron, filtered)
            
            scored = scorer.score(pron, filtered, sent)
            
            if scored:
                best = scored[0][0]
                confidence = scored[0][1]
                
                pron.misc = f"Antecedent={best['head_form']}"
                
                if RomaniMorphology.is_relative_pronoun(pron.form):
                    pron.lemma = best["head_lemma"]
                
                resolved_count += 1
                antecedents_found.append({
                    "pronoun": pron.form,
                    "antecedent": best["head_form"],
                    "sentence": sent_idx + 1,
                    "confidence": confidence
                })
                
                sent.text = " ".join(t.form for t in sent.tokens)
        
        dm.add_nps(nps, sent.sent_id)
    
    print(f"Всего предложений: {len(sentences)}")
    print(f"Найдено местоимений 3 лица: {total_pronouns_found}")
    print(f"Всего кандидатов рассмотрено: {total_candidates}")
    print(f"Разрешено местоимений: {resolved_count}")
    
    if total_pronouns_found > 0:
        print(f"Процент разрешения: {resolved_count/total_pronouns_found*100:.1f}%")
    
    if antecedents_found:
        print("Примеры разрешений:")
        for ex in antecedents_found[:15]:
            print(f"   • '{ex['pronoun']}' → '{ex['antecedent']}' (предл. {ex['sentence']}, conf={ex['confidence']})")
    else:
        print("НЕ РАЗРЕШЕНО НИ ОДНОГО МЕСТОИМЕНИЯ!")
    
    return sentences

def sentences_to_conll(sentences):
    conll_lines = []
    
    for sent in sentences:
        conll_lines.append(f"# text = {sent.text}")
        for tok in sent.tokens:
            line = "\t".join([
                str(tok.id),
                tok.form,
                tok.lemma if tok.lemma else "_",
                tok.pos if tok.pos else "_",
                getattr(tok, 'xpos', '_'),
                getattr(tok, 'feats', '_'),
                str(tok.head) if tok.head is not None else "_",
                tok.deprel if tok.deprel else "_",
                getattr(tok, 'deps', '_'),
                getattr(tok, 'misc', '_')
            ])
            conll_lines.append(line)
        conll_lines.append("")
    
    return conll_lines


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        conll_file = sys.argv[1]
    else:
        conll_file = "temp_rom.conll"    
    try:
        resolved_sentences = resolve_anaphora_rom(conll_file)
        for sent in resolved_sentences[:5]:
            # Показываем местоимения с антецедентами
            for tok in sent.tokens:
                if hasattr(tok, 'misc') and tok.misc and 'Antecedent' in tok.misc:
                    print(f"   ↳ {tok.form} → {tok.misc}")
        
        # Сохраняем результат
        output_file = "resolved_rom.conll"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences_to_conll(resolved_sentences)))
        
        print(f"Результат сохранен в {output_file}")
        
    except FileNotFoundError:
        print(f"Файл {conll_file} не найден")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()