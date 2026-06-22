import os
import re
from collections import Counter, defaultdict
from graph.higher_dim_graph import Graph

def get_connected_components(graph):
    adj = {v: set() for v in graph.vertices}
    for edge in graph.edges:
        adj[edge.agent_1].add(edge.agent_2)
        adj[edge.agent_2].add(edge.agent_1)
    
    visited = set()
    components = []
    for v in graph.vertices:
        if v not in visited:
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


def compare_graphs(en_graph, ru_graph, rom_graph, en_sentences, ru_sentences, rom_sentences):
    graphs = {'EN': en_graph, 'RU': ru_graph, 'ROM': rom_graph}
    sentences = {'EN': en_sentences, 'RU': ru_sentences, 'ROM': rom_sentences}
    
    results = {}
    
    for name, graph in graphs.items():
        sent_list = sentences[name]
        
        degree = defaultdict(int)
        for edge in graph.edges:
            degree[edge.agent_1] += 1
            degree[edge.agent_2] += 1
        
        edge_types = Counter()
        for edge in graph.edges:
            edge_types[edge.meaning] += 1
        
        pos_dist = Counter()
        for sent in sent_list:
            for tok in sent.tokens:
                pos_dist[tok.pos] += 1
        
        components = get_connected_components(graph)
        isolated = [c for c in components if len(c) == 1]
        giant = max(components, key=len) if components else []
        
        is_cyrillic = lambda w: any(ord(c) >= 0x0400 for c in w)
        latin_vertices = [v for v in graph.vertices if not is_cyrillic(v) and any(c.isalpha() for c in v)]
        
        top_vertices = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:20]
        top_edges = edge_types.most_common(10)
        
        bible_concepts = {
            'EN': {'god', 'lord', 'heaven', 'earth', 'light', 'darkness', 'water', 'sea',
                   'man', 'woman', 'adam', 'eve', 'spirit', 'day', 'night', 'son', 'father',
                   'abraham', 'isaac', 'jacob', 'noah', 'cain', 'abel', 'bless', 'create', 'covenant'},
            'RU': {'бог', 'господь', 'небо', 'земля', 'свет', 'тьма', 'вода', 'море',
                   'человек', 'женщина', 'адам', 'ева', 'дух', 'день', 'ночь', 'сын', 'отец',
                   'авраам', 'исаак', 'иаков', 'ной', 'каин', 'авель', 'благословить', 'сотворить', 'завет'},
            'ROM': {'дэвэл', 'рай', 'болыбэн', 'пхув', 'свэто', 'дуд', 'калыпэн', 'пани', 'мори', 'дэрьява',
                    'мануш', 'джувля', 'адамо', 'ева', 'духо', 'дывэс', 'рат', 'чхавэ', 'дад',
                    'авраамо', 'исако', 'яково', 'нои', 'каино', 'авелё', 'бахтякир', 'создыя'}
        }
        
        found_concepts = set()
        for v in graph.vertices:
            v_clean = v.replace('́', '').lower()
            for concept in bible_concepts[name]:
                if concept in v_clean:
                    found_concepts.add(concept)
        
        results[name] = {
            'vertices': len(graph.vertices),
            'edges': len(graph.edges),
            'density': len(graph.edges) / (len(graph.vertices) * (len(graph.vertices) - 1)) if len(graph.vertices) > 1 else 0,
            'avg_degree': sum(degree.values()) / len(graph.vertices) if graph.vertices else 0,
            'components': len(components),
            'isolated': len(isolated),
            'giant_size': len(giant),
            'giant_pct': len(giant) / len(graph.vertices) * 100 if graph.vertices else 0,
            'pos_dist': dict(pos_dist.most_common(5)),
            'top_vertices': top_vertices,
            'top_edges': top_edges,
            'latin_vertices': len(latin_vertices),
            'concepts_found': len(found_concepts),
            'concepts_list': sorted(found_concepts),
            'edge_type_count': len(edge_types),
        }
    
    print("1. БАЗОВЫЕ МЕТРИКИ")
    
    metrics_labels = [
        ('vertices', 'Вершин'),
        ('edges', 'Рёбер'),
        ('density', 'Плотность'),
        ('avg_degree', 'Средняя степень'),
        ('components', 'Компонент связности'),
        ('isolated', 'Изолированных вершин'),
        ('giant_pct', 'В гигантской комп. %'),
    ]
    
    for key, label in metrics_labels:
        values = {name: results[name][key] for name in graphs}
        if 'pct' in key or key == 'density':
            best = max(values, key=values.get)
            row = f"{label:<30} {values['EN']:<15.4f} {values['RU']:<15.4f} {values['ROM']:<15.4f} {best:<15}"
        elif key == 'components' or key == 'isolated':
            best = min(values, key=values.get)
            row = f"{label:<30} {values['EN']:<15} {values['RU']:<15} {values['ROM']:<15} {best:<15}"
        else:
            best = max(values, key=values.get)
            row = f"{label:<30} {values['EN']:<15} {values['RU']:<15} {values['ROM']:<15} {best:<15}"
        print(row)
    
    print("2. РАСПРЕДЕЛЕНИЕ ЧАСТЕЙ РЕЧИ")
    all_pos = set()
    for name in graphs:
        all_pos.update(results[name]['pos_dist'].keys())
    
    print(f"{'POS':<15}", end="")
    for name in graphs:
        print(f"{name:<20}", end="")
    print()
    print("-" * 75)
    
    for pos in sorted(all_pos, key=lambda p: sum(results[n]['pos_dist'].get(p, 0) for n in graphs), reverse=True)[:10]:
        print(f"{pos:<15}", end="")
        for name in graphs:
            count = results[name]['pos_dist'].get(pos, 0)
            total = sum(results[name]['pos_dist'].values())
            pct = count / total * 100 if total else 0
            print(f"{count:>5} ({pct:>5.1f}%)  ", end="")
        print()
    
    print("3. ТОП-10 ВЕРШИН ПО СТЕПЕНИ")
    for name in graphs:
        print(f"\n{name}:")
        print(f"  {'Вершина':<35} {'Степень':<10}")
        print(f"  {'-'*45}")
        for vertex, deg in results[name]['top_vertices'][:10]:
            print(f"  {vertex:<35} {deg:<10}")
    
    print("4. ТИПЫ ОТНОШЕНИЙ (ТОП-10)")
    for name in graphs:
        print(f"\n{name}:")
        print(f"  {'Отношение':<35} {'Кол-во':<10} {'Доля':<10}")
        print(f"  {'-'*55}")
        total_edges = results[name]['edges']
        for rel, count in results[name]['top_edges']:
            pct = count / total_edges * 100 if total_edges else 0
            print(f"  {rel:<35} {count:<10} {pct:<10.1f}%")
    
    print("5. КАЧЕСТВО МЕТОК ВЕРШИН")
    print(f"{'Показатель':<35} {'EN':<15} {'RU':<15} {'ROM':<15}")
    print(f"{'Вершин с латиницей':<35} {results['EN']['latin_vertices']:<15} {results['RU']['latin_vertices']:<15} {results['ROM']['latin_vertices']:<15}")
    
    for name in graphs:
        vertices = list(graphs[name].vertices)
        lengths = [len(v) for v in vertices]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        print(f"{'Средняя длина метки (' + name + ')':<35} {avg_len:<15.1f}")
    
    print("6. БИБЛЕЙСКИЕ КОНЦЕПТЫ")
    for name in graphs:
        print(f"\n{name}: найдено {results[name]['concepts_found']} концептов")
        concepts = results[name]['concepts_list'][:20]
        print(f"  {', '.join(concepts)}")
    
    print("7. СЕМАНТИЧЕСКОЕ СРАВНЕНИЕ")
    
    semantic_categories = {
        'Божественное': {
            'EN': ['god', 'lord', 'spirit', 'holy'],
            'RU': ['бог', 'господь', 'дух', 'свят'],
            'ROM': ['дэвэл', 'рай', 'духо', 'свэнто']
        },
        'Творение': {
            'EN': ['heaven', 'earth', 'light', 'darkness', 'water', 'sea', 'land'],
            'RU': ['небо', 'земля', 'свет', 'тьма', 'вода', 'море', 'суша'],
            'ROM': ['болыбэн', 'пхув', 'свэто', 'дуд', 'калыпэн', 'пани', 'мори', 'дэрьява']
        },
        'Человек': {
            'EN': ['man', 'woman', 'adam', 'eve', 'son', 'father', 'wife'],
            'RU': ['человек', 'женщина', 'адам', 'ева', 'сын', 'отец', 'жена'],
            'ROM': ['мануш', 'джувля', 'адамо', 'ева', 'чхавэ', 'дад', 'ромны']
        },
        'Действие': {
            'EN': ['create', 'make', 'say', 'see', 'give', 'bless', 'take', 'call'],
            'RU': ['сотворить', 'создать', 'сказать', 'увидеть', 'дать', 'благословить', 'взять', 'назвать'],
            'ROM': ['создыя', 'кэрдя', 'пхэндя', 'дыкхця', 'дыя', 'бахтякир', 'лыя', 'кхарэл']
        }
    }
    
    for cat_name, cat_words in semantic_categories.items():
        print(f"\n{cat_name}:")
        for name in graphs:
            found = []
            vertices = list(graphs[name].vertices)
            for word in cat_words[name]:
                for v in vertices:
                    if word in v.lower().replace('́', ''):
                        found.append(word)
                        break
            print(f"  {name}: найдено {len(found)}/{len(cat_words[name])} — {', '.join(found[:8])}")
    return results