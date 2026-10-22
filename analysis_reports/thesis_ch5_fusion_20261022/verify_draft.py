"""Audit thesis tables against existing results and user-reported counts."""
from pathlib import Path
import hashlib
import json
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
comparison_path = ROOT / 'analysis_reports/user_reference_distribution_20261022/comparison_results.json'
findings_path = ROOT / 'analysis_reports/thirteen_scene_ablation_v5_20261021/numerical_findings.json'
comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
findings = json.loads(findings_path.read_text(encoding='utf-8'))
composition_path = HERE / 'priority_composition_data_and_provenance.json'
composition = json.loads(composition_path.read_text(encoding='utf-8'))
deviation_path = HERE / 'reference_deviation_data_and_provenance.json'
deviation = json.loads(deviation_path.read_text(encoding='utf-8'))
assert not deviation['accuracy_computed'] and not deviation['synthetic_demo_used']
configuration_rows = {row['variant']: row for row in composition['summary_rows']}
draft = next(HERE.glob('5.3_*.md'))
content = draft.read_text(encoding='utf-8')
counts = [960, 1165, 1185, 4710, 1459, 2731, 881, 4644, 2625, 4631, 11839, 6454, 14084]
checked_configurations = set()
checked_scenes = set()
for line in content.splitlines():
    if not line.startswith('|'):
        continue
    fields = [field.strip().replace('**', '') for field in line.strip().strip('|').split('|')]
    if len(fields) == 5 and fields[0] in configuration_rows:
        row = configuration_rows[fields[0]]
        conflict = row['NC_conflict_percent']
        assert fields[1:] == [f"{row['object_unresolved_percent']:.2f}",
                              f"{row['unknown_function_percent']:.2f}",
                              f'{conflict:.2f}' if conflict is not None else '—',
                              f"{row['macro_tvd_pp']:.2f}"], fields
        checked_configurations.add(fields[0])
    if len(fields) == 6 and re.fullmatch(r'M\d+', fields[0]):
        index = int(fields[0][1:]) - 1
        scene = comparison['models'][index]
        n = scene['geometry_instances']
        assert fields[1:] == [f'{n:,}', f'{counts[index]:,}', f'{100*counts[index]/n:.2f}',
                               f"{scene['tvd_percentage_points']['NCG_H0']:.2f}",
                               f"{scene['tvd_percentage_points']['NCG_H1']:.2f}"], fields
        checked_scenes.add(fields[0])

assert len(checked_configurations) == 14 and len(checked_scenes) == 13
assert content.count('**图5-1 ') == content.count('**图5-2 ') == 1
assert '### 5.3.4 ' not in content
section = '### 5.3.2 ' + content.split('### 5.3.2 ', 1)[1].split('### 5.3.3 ', 1)[0]
images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', section)
assert len(images) == 2 and all((HERE / image).is_file() for image in images)
assert section.count('**表5-5 ') == 1
excerpt = HERE / '5.3.2_一表两图正文.md'
excerpt.write_text(section.rstrip()+'\n', encoding='utf-8')
population = sum(scene['geometry_instances'] for scene in comparison['models'])
assert population == findings['nodes'] == 68972
assert sum(counts) == 57368 and population - sum(counts) == 11604
assert findings['tree']['priority_changed'] == findings['tree']['priority_raised'] == 11449
assert findings['tree']['priority_lowered'] == 0
for filename, digest in comparison['source_sha256'].items():
    assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest

audit = dict(
    status='ARITHMETIC_AND_SOURCE_TABLE_CHECKS_PASSED',
    checked_configuration_rows=14, checked_scene_rows=13,
    valid_instances=population, user_reported_correct=sum(counts),
    user_reported_incorrect=population-sum(counts),
    micro_accuracy_from_user_counts_percent=100*sum(counts)/population,
    macro_scene_accuracy_from_user_counts_percent=statistics.mean(
        100*c/scene['geometry_instances'] for c, scene in zip(counts, comparison['models'])),
    human_evaluation_source='User-provided NCG_H1-versus-manual summary in conversation',
    human_annotation_records_independently_verified=False,
    evaluation_definition_in_draft='Exact P-label agreement; confirm this matches original evaluation',
    accuracy_available_for_other_configurations=False,
    synthetic_demo_used=False,
    source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (comparison_path, findings_path, composition_path, deviation_path)},
    original_model_report_sha256=comparison['source_sha256'],
    draft_sha256=hashlib.sha256(draft.read_bytes()).hexdigest(),
    section_excerpt_sha256=hashlib.sha256(excerpt.read_bytes()).hexdigest(),
    section_tables=1, section_figures=2,
)
(HERE / 'writing_data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: audit[key] for key in ('status', 'checked_configuration_rows', 'checked_scene_rows',
    'micro_accuracy_from_user_counts_percent', 'macro_scene_accuracy_from_user_counts_percent',
    'human_annotation_records_independently_verified', 'synthetic_demo_used')}))
