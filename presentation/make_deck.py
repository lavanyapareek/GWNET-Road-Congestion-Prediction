"""Progress deck (15 Sept 2026): progress, results, plan to the 21 Sept demo, work split. Run after make_charts.py."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'data' / 'traffic' / 'congestion' / 'results'
CH = HERE / 'charts'
final = {r: json.load(open(RES / f'{r}_final.json'))['final'] for r in ('R0', 'A1', 'A3')}
R0, A1, A3 = (final[k]['test']['avg'][0] for k in ('R0', 'A1', 'A3'))

INK, INK2, MUTED, LINE = RGBColor(0x0b, 0x0b, 0x0b), RGBColor(0x52, 0x51, 0x4e), RGBColor(0x89, 0x87, 0x81), RGBColor(0xe1, 0xe0, 0xd9)
BLUE, PALE, SOFT = RGBColor(0x2a, 0x78, 0xd6), RGBColor(0xee, 0xf4, 0xfc), RGBColor(0xf6, 0xf6, 0xf3)
FONT = 'Arial'                                       # on every Mac and Windows machine
SCALE = 0.92                                         # Arial runs ~10% wider than Calibri

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N_SLIDES = 12


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP):
    """paras: list of (text, size, bold, color[, align]) or str."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = Inches(0.02)
    for i, p in enumerate(paras):
        if isinstance(p, str):
            p = (p, 16, False, INK)
        t, size, bold, color = p[:4]
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p[4] if len(p) > 4 else PP_ALIGN.LEFT
        para.space_after = Pt(size * 0.45)
        run = para.add_run()
        run.text = t
        run.font.size, run.font.bold, run.font.color.rgb, run.font.name = Pt(size * SCALE), bold, color, FONT
    return tb


def bullets(slide, x, y, w, h, items, size=16):
    """items: (lead, rest) pairs rendered as '• lead rest' with the lead in bold."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (lead, rest) in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(size * 0.7)
        for t, bold, color in (('•  ', False, BLUE), (lead, True, INK), (rest, False, INK2)):
            r = para.add_run()
            r.text, r.font.size, r.font.bold, r.font.color.rgb, r.font.name = t, Pt(size * SCALE), bold, color, FONT
    return tb


def base(title, eyebrow, n, notes):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.6, 0.35, 12, 0.35, [(eyebrow.upper(), 11, True, MUTED)])
    text(s, 0.6, 0.62, 12.1, 0.9, [(title, 28, True, INK)])
    ln = s.shapes.add_connector(1, Inches(0.6), Inches(7.0), Inches(12.73), Inches(7.0))
    ln.line.color.rgb, ln.line.width = LINE, Pt(0.75)
    text(s, 0.6, 7.05, 9, 0.3, [('Freeway congestion forecasting · LargeST San Diego 2019 · progress review 15 Sept 2026', 10, False, MUTED)])
    text(s, 11.9, 7.05, 0.83, 0.3, [(f'{n} / {N_SLIDES}', 10, False, MUTED, PP_ALIGN.RIGHT)])
    s.notes_slide.notes_text_frame.text = notes
    return s


def card(slide, x, y, w, h, head, body, fill=SOFT, head_color=INK):
    r = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    r.adjustments[0] = 0.08
    r.fill.solid()
    r.fill.fore_color.rgb = fill
    r.line.fill.background()
    r.shadow.inherit = False
    text(slide, x + 0.2, y + 0.15, w - 0.4, h - 0.3, [(head, 17, True, head_color)] + [(b, 13.5, False, INK2) for b in body])
    return r


def table(slide, x, y, w, rows, col_w, size=13, header_fill=RGBColor(0xee, 0xee, 0xe9), row_h=0.42):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    t = shp.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = header_fill if i == 0 else RGBColor(0xff, 0xff, 0xff)
            c.margin_left = c.margin_right = Inches(0.08)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            tf.paragraphs[0].text = ''
            r = tf.paragraphs[0].add_run()
            r.text = str(val)
            r.font.size, r.font.name = Pt(size * SCALE), FONT
            r.font.bold = i == 0
            r.font.color.rgb = INK if i == 0 or j == 0 else INK2
    return t


# 1 ── title
s = prs.slides.add_slide(BLANK)
text(s, 0.9, 1.7, 11.5, 0.4, [('CAPSTONE · PROGRESS REVIEW · 15 SEPTEMBER 2026', 13, True, MUTED)])
text(s, 0.9, 2.2, 11.5, 2.0, [('Forecasting freeway congestion in San Diego', 46, True, INK)])
text(s, 0.9, 3.95, 11.5, 1.0, [('Graph neural network on 716 Caltrans sensors — flow and congestion, 15 minutes to 3 hours ahead', 20, False, INK2)])
text(s, 0.9, 5.3, 11.5, 0.5, [('Team of 5  ·  Progress so far, results, and the plan to the live demo on 21 September', 15, False, MUTED)])
s.notes_slide.notes_text_frame.text = ('Today: where we are, what the results say so far, and how the five of us will get to the live demo '
                                      'on the 21st. About 12 minutes.')

# 2 ── what we are building
s = base('What we are building', 'Goal', 2,
         'Forecast traffic flow and a congestion level for every sensor, 15 min to 3 h ahead. The baseline to beat is our '
         'LSTM (average MAE 26.35). Three deliverables for the 21st: the model, a dashboard, and a chatbot that answers '
         'questions from the model\'s own predictions.')
text(s, 0.6, 1.55, 12, 0.6, [('716 freeway sensors · San Diego (Caltrans District 11) · all of 2019 at 15-minute resolution · '
                              'forecasts 15 min → 3 h ahead', 16, False, INK2)])
card(s, 0.6, 2.35, 3.9, 3.9, 'Model', ['Graph neural network (GWNet) that', 'predicts flow and congestion', '(free / heavy / congested) for',
                                         'every sensor, 12 steps ahead.', '', 'Target: beat our LSTM baseline', f'(MAE 26.35).'])
card(s, 4.72, 2.35, 3.9, 3.9, 'Dashboard', ['Streamlit app: congestion map with', 'horizon and time sliders, sensor', 'drill-down, model results.', '',
                                              'Replays the Oct–Dec 2019 test', 'period as if live.'])
card(s, 8.84, 2.35, 3.9, 3.9, 'Chatbot', ['Claude (Anthropic API) with tools', 'that query our predictions and', 'results — every number it quotes', 'is checkable.', '',
                                            '"Which stretches jam at 5 pm?"', '"Why was Christmas hard?"'])

# 3 ── progress so far
s = base('Progress so far', 'Where we are', 3,
         'Everything above the last line is done and checked. The data audit reproduced the published benchmark exactly, '
         'so our numbers are comparable to the literature. We then trained the main model and the graph ablation, got '
         'Caltrans PeMS data for real congestion labels and incidents, built the feature pipeline, and started the '
         'training queue that runs until Thursday.')
steps = [('Data audit', 'dataset identical to the published LargeST benchmark; zeros, 999s, time zones checked'),
         ('Baselines', 'last-value, time-of-day profile, our LSTM (MAE 26.35)'),
         ('Main model (R0)', f'GWNet trained on the M2 laptop: test MAE {R0:.2f}'),
         ('Graph ablation', f'road graph only {A1:.2f}, no graph {A3:.2f} — the graph drives the gain'),
         ('PeMS data', 'occupancy for 716 sensors + 49,198 CHP incidents; matches LargeST 99.98%'),
         ('Congestion label', 'occupancy-based, validated (bottlenecks, holidays, incidents)'),
         ('Feature pipeline', 'data fixes, lanes/time, holidays/school, weather, incidents — all tested'),
         ('Training queue', 'feature rows R2 → R6 + two runs, back to back until Thu 17 Sept')]
for i, (lead, rest) in enumerate(steps):
    y = 1.6 + i * 0.64
    done = i < len(steps) - 1
    mark = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.75), Inches(y + 0.07), Inches(0.3), Inches(0.3))
    mark.fill.solid()
    mark.fill.fore_color.rgb = BLUE if done else RGBColor(0xff, 0xff, 0xff)
    mark.line.color.rgb = BLUE
    mark.line.width = Pt(2)
    if done:
        mark.text_frame.text = '✓'
        p = mark.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.runs[0].font.size, p.runs[0].font.bold, p.runs[0].font.color.rgb = Pt(11), True, RGBColor(0xff, 0xff, 0xff)
        p.runs[0].font.name = FONT
    text(s, 1.3, y, 11.3, 0.5, [(lead + ('' if done else '  — in progress'), 16, True, INK)])
    text(s, 4.2, y + 0.03, 8.5, 0.5, [(rest, 14.5, False, INK2)])

# 4 ── data findings
s = base('What the data audit taught us', 'Findings', 4,
         'Four things from the audit. First, our data is exactly the benchmark, so results are comparable. Second, zeros '
         'are sensor outages, not empty roads. Third, the test period contains Thanksgiving and Christmas, which break '
         'normal patterns. Fourth, PeMS gave us occupancy, which is what congestion actually is, plus incidents.')
bullets(s, 0.6, 1.6, 12.1, 5.2, [
    ('Our data is the benchmark. ', 'The last-value baseline reproduces the published LargeST numbers to 4 decimals, '
                                    'so every result is directly comparable to the paper.'),
    ('Zeros are outages, not empty roads. ', '94% of zero readings sit in outages longer than a day; neighbouring '
                                             'sensors carry normal traffic meanwhile. We mask and fill them.'),
    ('Holidays drive test-period error. ', 'Thanksgiving week and Dec 21–31 explain ~75% of the jump in error from '
                                           'validation to test; Christmas Day is 5.5× a normal day.'),
    ('PeMS gives us real congestion data. ', 'Occupancy (the standard congestion signal) for all 716 sensors, matching '
                                             'LargeST flow 99.98%; 49,198 incidents, 92.6% matched to sensors.'),
    ('Local compute works. ', 'Everything trains on a MacBook Air M2 (8 GB): ~5 h per model, checkpointed every '
                              '~5 minutes and resumable.')], size=17)

# 5 ── GWNet vs LSTM
s = base('GWNet beats our LSTM — most at long horizons', 'Result 1', 5,
         f'Test error by forecast horizon. GWNet R0 averages {R0:.2f} against the LSTM\'s 26.35, 23% better. The gap grows '
         'with horizon: at 3 hours 37.5 drops to 25.3. The paper\'s GWNet reaches 17.74 with about 80 epochs on a '
         'data-centre GPU; we used 16 epochs on a laptop.')
s.shapes.add_picture(str(CH / 'horizon.png'), Inches(0.6), Inches(1.5), height=Inches(5.2))
text(s, 8.95, 1.75, 3.9, 1.3, [(f'{R0:.2f}', 44, True, BLUE), ('average test MAE (GWNet R0)', 14, False, INK2)])
text(s, 8.95, 3.2, 3.9, 1.3, [(f'−{(1 - R0 / 26.35) * 100:.0f}%', 44, True, INK), ('vs our LSTM (26.35)', 14, False, INK2)])
text(s, 8.95, 4.65, 3.9, 1.6, [('−32% at 3 h', 30, True, INK), ('error at 3 hours ahead: 37.5 → 25.3', 14, False, INK2),
                                ('paper GWNet 17.74 (≈80 epochs)', 14, False, MUTED)])

# 6 ── graph ablation
s = base('The gain comes from the graph', 'Result 2', 6,
         f'Same model with parts removed. Without any graph (A3) it is no better than the LSTM: {A3:.2f}. The road network '
         f'alone (A1) gets {A1:.2f}, about two-thirds of the benefit; the learned connections between sensors add the rest '
         f'({R0:.2f}). So spatial information — traffic upstream and downstream — is what makes the difference.')
s.shapes.add_picture(str(CH / 'models_bar.png'), Inches(0.5), Inches(1.55), width=Inches(7.9))
bullets(s, 8.7, 1.8, 4.1, 5, [
    ('No graph = LSTM. ', f'Removing the graph ({A3:.2f}) loses the whole advantage.'),
    ('Road network: ~⅔ of the gain. ', f'Road graph only reaches {A1:.2f}.'),
    ('Learned links: the rest. ', f'Adding them gives {R0:.2f}. One more run (A2) separates the two.'),
    ('Gap to the paper is budget. ', '16 epochs on a laptop vs ~80 on an A6000.')], size=15)

# 7 ── labels
s = base('Congestion must be measured with occupancy, not flow', 'Result 3', 7,
         'A key methodological finding. A congestion label built from flow alone calls quiet holiday roads jammed: on '
         'Christmas Day it flags 47% of the network, while occupancy shows almost no congestion. So we built the label '
         'from PeMS occupancy and validated it.')
s.shapes.add_picture(str(CH / 'labels.png'), Inches(0.5), Inches(1.55), width=Inches(7.9))
bullets(s, 8.7, 1.8, 4.1, 5, [
    ('Flow-only fails. ', 'Precision 7%, recall 3% against occupancy; 47% "congested" on Christmas vs 0.03%.'),
    ('Occupancy label holds up. ', '~5.5% congested, peaks at 7 am and 4 pm, known bottlenecks on top.'),
    ('Hourly is close enough. ', '96% agreement with 15-min occupancy; catches 90% of congestion.'),
    ('Incidents show up. ', 'Congestion near a collision jumps +14 points at its start hour.')], size=15)

# 8 ── training queue
s = base('Training now: one run per feature group', 'In progress', 8,
         'The queue runs back to back on the laptop. Each feature row adds one group of inputs on top of the previous '
         'one, all with the congestion head, so the final table shows what each group is worth. Everything checkpoints '
         'and resumes automatically.')
table(s, 0.6, 1.6, 12.1, [
    ['Run', 'What it adds', 'Why', 'Expected done'],
    ['R2', 'Missing-data mask + fill', 'Stop outages looking like empty roads', 'Tue 15 · evening'],
    ['A2', 'Learned links only (no road graph)', 'Completes the graph ablation', 'Tue 15 · night'],
    ['R1', 'R0 + congestion head', 'Baseline for all feature rows', 'Wed 16 · early morning'],
    ['R3', '+ lanes, cyclical time, sensor attributes', 'Makes sensors comparable', 'Wed 16 · late morning'],
    ['R4', '+ holidays, school calendar', 'Targets the holiday-week errors', 'Wed 16 · evening'],
    ['R5', '+ weather (5 stations)', 'Rain effect: ~147 rain hours in test', 'Wed 16 · night'],
    ['R6', '+ incidents (CHP)', 'Biggest expected gain for congestion', 'Thu 17 · morning']],
    [0.9, 4.2, 4.6, 2.4], size=14, row_h=0.52)
text(s, 0.6, 6.0, 12, 0.6, [('~5 h per run on the M2 · new runs start only on mains power · results table and charts update as runs finish',
                              13.5, False, MUTED)])

# 9 ── demo architecture
s = base('What we will demo on 21 September', 'Plan', 9,
         'Left to right: the trained models produce predictions for the whole test period, stored once. The Streamlit app '
         'reads them for the map, the sensor view and the results pages. The chatbot uses Claude through the Anthropic '
         'API with a small set of tools that read the same predictions, so it cannot invent numbers. The demo replays '
         'Oct–Dec 2019 as if live; a live feed would need a Caltrans real-time source and is future work.')
boxes = [(0.6, 'Data & models', ['LargeST flow · PeMS occupancy', 'weather · CHP incidents', 'GWNet R0–R6 (trained)']),
         (3.95, 'Prediction store', ['every sensor × 15-min step', 'flow + congestion class', 'for Oct–Dec 2019 (test)']),
         (7.3, 'Streamlit dashboard', ['congestion map (OSM tiles)', 'horizon + time sliders', 'sensor drill-down · results'])]
for x, head, body in boxes:
    card(s, x, 1.75, 3.0, 2.3, head, body, fill=PALE, head_color=BLUE)
for x in (3.62, 6.97):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(2.72), Inches(0.3), Inches(0.35))
    a.fill.solid()
    a.fill.fore_color.rgb = MUTED
    a.line.fill.background()
card(s, 10.65, 1.75, 2.1, 2.3, 'Chatbot', ['Claude API', 'tool use', 'in the app'], fill=PALE, head_color=BLUE)
a = s.shapes.add_shape(MSO_SHAPE.LEFT_RIGHT_ARROW, Inches(10.32), Inches(2.72), Inches(0.33), Inches(0.35))
a.fill.solid()
a.fill.fore_color.rgb = MUTED
a.line.fill.background()
text(s, 0.6, 4.35, 6, 0.4, [('Chatbot tools (answers only from our data)', 16, True, INK)])
table(s, 0.6, 4.8, 12.1, [
    ['Tool', 'Answers questions like'],
    ['get_forecast(sensor or road, time, horizon)', '"What will I-5 northbound look like at 5 pm?"'],
    ['top_hotspots(time, horizon)', '"Where will congestion be worst in the next hour?"'],
    ['compare_models(metric, stratum)', '"How much does weather help on rainy days?"']],
    [5.0, 7.1], size=13, row_h=0.42)

# 10 ── timeline
s = base('Timeline to 21 September', 'Plan', 10,
         'Training finishes Thursday morning. Dashboard and chatbot start tomorrow (Wednesday) in parallel, first against '
         'the models that are already done, then switched to the final ones. Evaluation Thursday–Friday; integration '
         'and the deck Saturday–Sunday; rehearsal Sunday; demo Monday the 21st.')
s.shapes.add_picture(str(CH / 'gantt.png'), Inches(0.85), Inches(1.6), width=Inches(11.6))
text(s, 0.6, 6.35, 12, 0.5, [('M1–M5 = team members (next slide). Dashboard and chatbot build against the finished models (R0, A1, A3) '
                              'first, then switch to the final ones.', 13.5, False, MUTED)])

# 11 ── who does what
s = base('Who does what', 'Work split', 11,
         'Five workstreams, one owner each, with named deliverables and dates. Rename the members to the actual people. '
         'Daily 15-minute stand-up; integration starts Friday.')
table(s, 0.6, 1.55, 12.1, [
    ['Member', 'Workstream', 'Deliverables', 'Due'],
    ['M1', 'ML & evaluation', 'Monitor training queue; final ablation table; rain / incident / holiday / peak breakdowns; '
                              'confidence intervals', 'Fri 18'],
    ['M2', 'Data & predictions', 'Export predictions for every model; sensor map layer; incident & weather overlays; '
                                 'data documentation', 'Fri 18'],
    ['M3', 'Dashboard (Streamlit)', 'Congestion map with horizon/time sliders; sensor drill-down; results pages', 'Sat 19'],
    ['M4', 'Chatbot (Claude API)', 'Tool design (forecast, hotspots, model comparison); prompt & guardrails; test set of '
                                   '30 questions', 'Sat 19'],
    ['M5', 'Integration, QA & story', 'End-to-end testing; demo script; backup video; final deck; limitations section',
     'Sun 20']], [1.0, 2.4, 7.3, 1.4], size=14, row_h=0.75)

# 12 ── risks
s = base('Risks and how we handle them', 'Risks', 12,
         'The main risk is compute: one laptop runs everything. The queue is ordered so the most important runs finish '
         'first, and the demo works with whatever has finished. Live data is out of scope, so we replay 2019. The chatbot '
         'only answers from tools, and we record a backup video of the demo.')
table(s, 0.6, 1.55, 12.1, [
    ['Risk', 'Impact', 'Mitigation'],
    ['One 8 GB laptop trains every model', 'Late runs if it sleeps or unplugs', 'Ordered queue; resumes from checkpoints; '
                                                                             'demo uses finished models'],
    ['No public live Caltrans traffic feed', 'Cannot show today\'s traffic', 'Replay the 2019 test period; live feed is '
                                                                           'future work'],
    ['Chatbot states wrong numbers', 'Loss of credibility', 'Answers only via tools over our predictions; 30-question '
                                                            'test set'],
    ['API key / cost for Claude', 'Chatbot blocked', 'Set up key by Wed 16; low usage; fallback: scripted Q&A'],
    ['Live demo fails on the day', 'Weak finish', 'Recorded backup video; local run; offline map fallback']],
    [3.8, 3.2, 5.1], size=14, row_h=0.72)

out = HERE / 'Progress_Review_2026-09-15.pptx'
prs.save(out)
print('saved', out, '|', len(prs.slides), 'slides')
