"""Register the four supplied papers plus Sulawesi, then verify the real API flow.

Run with the backend's existing Python. State is saved after each acquisition so
repeating the command reuses registered files rather than duplicating uploads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', required=True)
    parser.add_argument('--papers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workbench-api', default='https://tapa-research-workbench-backend-production.up.railway.app/api/v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'pilot.json'
    state = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': []}

    def save():
        manifest_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    def call(path, body=None, raw=None, base=None):
        headers = {'Accept': 'application/json'}
        if body is not None:
            raw = json.dumps(body, ensure_ascii=False).encode()
            headers['Content-Type'] = 'application/json'
        request = Request((base or args.api).rstrip('/') + path, data=raw, headers=headers)
        with urlopen(request, timeout=120) as response:
            return json.load(response)

    # Wait for the intended implementation, rather than uploading to the old API.
    for _ in range(90):
        try:
            caps = call('/research-runs/capabilities')
            if 'span_id' in caps.get('evidence_selections', []):
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        raise RuntimeError('Updated staging API did not become ready')
    print('Updated staging API ready', flush=True)
    if 'material_id' not in state:
        material = call('/repository/materials', {'title': 'Watanabe research pilot — four Chinese papers and Sulawesi',
            'source_type': 'publication', 'collection': 'Watanabe POC 2026-10-19', 'status': 'ready_for_text_extraction'})
        state['material_id'] = material['id']; save()
    material_id = state['material_id']
    sources = sorted(args.papers.glob('*.pdf'))
    if len(sources) != 4:
        raise RuntimeError('Expected all four supplied Chinese PDFs')
    inventory = call('/repository/materials?limit=100', base=args.workbench_api)['materials']
    sulawesi = next(row for row in inventory if row['title'] == 'Sulawesi beaters over time')
    details = call('/repository/materials/' + sulawesi['id'], base=args.workbench_api)
    source_file = details['files'][0]
    sulawesi_path = args.output / source_file['original_filename']
    if not sulawesi_path.exists():
        with urlopen(args.workbench_api + f"/repository/materials/{sulawesi['id']}/files/{source_file['id']}", timeout=120) as response:
            sulawesi_path.write_bytes(response.read())
    sources.append(sulawesi_path)
    for path in sources:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if any(row['sha256'] == sha for row in state['files']):
            continue
        uploaded = call(f'/repository/materials/{material_id}/files?' + urlencode({'filename': path.name}), raw=path.read_bytes())
        state['files'].append({'filename': path.name, 'sha256': sha, 'file_id': uploaded['id']})
        save(); print('Registered: ' + path.name, flush=True)
    expected_files = len(state['files'])
    previous = None
    for _ in range(180):
        prep = call(f'/repository/materials/{material_id}/processing')
        runs = call('/research-runs?' + urlencode({'material_id': material_id, 'limit': 20}))['items']
        candidates = [run for run in runs if run.get('document_count') == expected_files]
        pinned = (state.get('run') or {}).get('run_id')
        selected_id = pinned or (candidates[0]['run_id'] if candidates else None)
        run = call('/research-runs/' + selected_id) if selected_id else None
        summary = {'preparation': (prep.get('job') or {}).get('status'),
                   'stage': (prep.get('job') or {}).get('stage'),
                   'graph': run.get('status') if run else None,
                   'coverage': prep['coverage'].get('embedded_segment_count')}
        if summary != previous:
            print(json.dumps(summary), flush=True); previous = summary
        state['preparation'] = prep; state['run'] = run; save()
        graph_file = None
        if run:
            artifacts = run.get('artifacts', [])
            graph_file = 'graph_input.json' if 'graph_input.json' in artifacts else 'validated_preview.json' if 'validated_preview.json' in artifacts else None
        if graph_file and prep['coverage'].get('embedded_segment_count', 0) > 0:
            graph = call(f"/research-runs/{run['run_id']}/artifacts/{graph_file}")
            if not graph.get('entities') or not graph.get('assertions'):
                time.sleep(5); continue
            entities = {entity['entity_id'] for entity in graph['entities']}
            evidence = {item['evidence_id']: item for item in graph['evidence']}
            batches = call(f"/research-runs/{run['run_id']}/artifacts/primary_batches.json")
            units = {unit['unit_id']: unit for batch in batches for unit in batch['units']}
            for item in evidence.values():
                unit = units[item['unit_id']]
                assert item['source_id'] == unit['source_id']
                assert item['source_version_id'] == unit['source_version_id']
                assert hashlib.sha256(unit['text'].encode()).hexdigest() == item['unit_text_sha256']
                assert hashlib.sha256(item['text'].encode()).hexdigest() == item['text_sha256']
                spans = item['character_locator']['spans']
                assert spans and all(unit['text'][span['start']:span['end']] == item['text'] for span in spans)
            for entity in graph['entities']:
                assert entity['evidence_ids'] and all(key in evidence and evidence[key]['source_id'] == entity['source_id'] for key in entity['evidence_ids'])
            for claim in graph['assertions']:
                assert claim['subject_id'] in entities
                assert not claim.get('object_id') or claim['object_id'] in entities
                assert claim['evidence_ids'] and all(item in evidence for item in claim['evidence_ids'])
            exact = call('/repository/search?' + urlencode({'q': '石拍', 'material_id': material_id, 'limit': 5}))
            related = call('/repository/ai/semantic-search', {'query': '石拍', 'material_id': material_id, 'limit': 5})
            assert exact['results'], 'Chinese exact search returned no evidence'
            assert related['results'], 'Semantic search returned no evidence'
            (args.output / 'graph.json').write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding='utf-8')
            (args.output / 'primary-batches-baseline.json').write_text(json.dumps(batches, ensure_ascii=False, indent=2), encoding='utf-8')
            state['verification'] = {'sources': len(graph['sources']), 'entities': len(entities),
                'relationships': len(graph['assertions']), 'quotations': len(evidence),
                'exact_quote_offset_checks': len(evidence),
                'exact_search_hits': len(exact['results']), 'semantic_search_hits': len(related['results']),
                'partial_extraction': bool(graph.get('partial_extraction')), 'artifact': graph_file,
                'scope': 'All five PDFs registered; full native/OCR ingestion. Semantic completeness is reported by run status.'}
            state['browser_url'] = f"http://127.0.0.1:5173/?view=knowledge&material={material_id}&graph={run['run_id']}"
            save(); print('VERIFIED ' + json.dumps(state['verification']), flush=True)
            print(state['browser_url'], flush=True)
            return
        if prep.get('job', {}).get('status') == 'error' and not run:
            raise RuntimeError('Preparation failed: ' + str(prep['job'].get('detail')))
        time.sleep(5)
    raise RuntimeError('Pilot is still processing. Rerun to resume monitoring saved IDs.')


if __name__ == '__main__':
    main()
