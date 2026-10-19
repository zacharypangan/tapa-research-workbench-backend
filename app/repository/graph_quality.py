"""Structural connectivity measures; never an extraction-accuracy score."""
from collections import Counter


def connectivity_report(graph):
    entities = {row['entity_id']: row for row in graph.get('entities', [])}
    sources = {row['source_id']: row for row in graph.get('sources', [])}

    def origin(source_id):
        source = sources.get(source_id, {})
        if source.get('origin') == 'web' or source.get('source_kind') in {'web', 'html', 'web_page', 'web_snapshot'}:
            return 'web'
        if source.get('origin') in {'primary', 'document'} or str(source.get('source_locator', '')).startswith('workbench:material/'):
            return 'document'
        return 'other'

    def measure(rows):
        parents = {key: key for key in entities}
        connected, pairs = set(), set()
        cross_source = document_web = claims = negative = uncertain = 0

        def find(key):
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key

        for row in rows:
            a, b = row.get('subject_id'), row.get('object_id')
            if a not in entities or b not in entities:
                continue
            claims += 1
            negative += row.get('polarity') == 'negative'
            uncertain += row.get('polarity') == 'uncertain'
            pairs.add((a, row.get('predicate'), b))
            connected.update([a, b])
            parents[find(a)] = find(b)
            sa, sb = entities[a].get('source_id'), entities[b].get('source_id')
            cross_source += bool(sa and sb and sa != sb)
            document_web += {origin(sa), origin(sb)} == {'document', 'web'}
        components = Counter(find(key) for key in entities)
        mixed_components = set()
        origins = {}
        for key, entity in entities.items():
            origins.setdefault(find(key), set()).add(origin(entity.get('source_id')))
        for component, roles in origins.items():
            if {'document', 'web'} <= roles:
                mixed_components.add(component)
        return {'relationship_claims': claims, 'distinct_directed_connections': len(pairs),
                'concepts_with_edges': len(connected), 'isolated_concepts': len(entities) - len(connected),
                'components': len(components), 'largest_component': max(components.values(), default=0),
                'cross_source_claims': cross_source, 'document_web_claims': document_web,
                'concepts_in_document_web_components': sum(components[key] for key in mixed_components),
                'negative_claims': negative, 'uncertain_claims': uncertain}

    reported = graph.get('assertions', []) + graph.get('supported_relationships', [])
    combined = reported + graph.get('alignments', []) + graph.get('inferences', [])
    return {'method_version': 'graph-connectivity/1', 'reported': measure(reported), 'combined': measure(combined),
            'literal_claims': sum(not row.get('object_id') for row in graph.get('assertions', [])),
            'recovered_relationships': len(graph.get('supported_relationships', [])),
            'interpretation': 'Counts describe saved structure, including negative and uncertain links; they do not measure truth, source independence, extraction completeness or corroboration. Literal attributes and scope references are not traversable concept relationships.'}
