import unittest
from app.repository.graph_quality import connectivity_report


class ConnectivityTests(unittest.TestCase):
    def test_literals_and_context_do_not_invent_connectivity(self):
        graph = {'entities': [{'entity_id': key} for key in ['a', 'b', 'c']],
                 'assertions': [{'subject_id': 'a', 'value': 'definition', 'scope': {'place_id': 'b'}}]}
        report = connectivity_report(graph)
        self.assertEqual(report['reported']['concepts_with_edges'], 0)
        self.assertEqual(report['reported']['components'], 3)
        self.assertEqual(report['literal_claims'], 1)

    def test_separate_reported_proposals_and_mixed_components(self):
        graph = {'sources': [{'source_id': 'paper', 'origin': 'primary'}, {'source_id': 'site', 'origin': 'web'}],
                 'entities': [{'entity_id': 'a', 'source_id': 'paper'}, {'entity_id': 'b', 'source_id': 'paper'}, {'entity_id': 'c', 'source_id': 'site'}],
                 'assertions': [{'subject_id': 'a', 'object_id': 'b', 'predicate': 'uses', 'polarity': 'negative'}],
                 'inferences': [{'subject_id': 'b', 'object_id': 'c', 'predicate': 'possible_match', 'polarity': 'uncertain'}]}
        report = connectivity_report(graph)
        self.assertEqual(report['reported']['concepts_with_edges'], 2)
        self.assertEqual(report['reported']['negative_claims'], 1)
        self.assertEqual(report['reported']['document_web_claims'], 0)
        self.assertEqual(report['combined']['document_web_claims'], 1)
        self.assertEqual(report['combined']['concepts_in_document_web_components'], 3)
        self.assertEqual(report['combined']['concepts_with_edges'], 3)

    def test_parallel_claims_count_once_as_a_visual_triple(self):
        edge = {'subject_id': 'a', 'object_id': 'b', 'predicate': 'uses'}
        graph = {'entities': [{'entity_id': 'a'}, {'entity_id': 'b'}], 'assertions': [edge, edge.copy()]}
        report = connectivity_report(graph)['reported']
        self.assertEqual(report['relationship_claims'], 2)
        self.assertEqual(report['distinct_directed_connections'], 1)


if __name__ == '__main__':
    unittest.main()
