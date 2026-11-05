"""Pure, bounded prompt views of complete private company/memory evidence.

The original documents, source identifiers, retrieval ranks and feedback stay
outside these views. Short references are deterministic and locally reversible
through the separately returned reference maps; they carry no extra evidence.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

from .calendar import aware
from .registry import TICKERS


def _selected(tickers):
    if tickers is None: return tuple(TICKERS)
    if isinstance(tickers, (str, bytes)) or not isinstance(tickers, Sequence):
        raise ValueError('An explicit ordered pair or all ten eligible tickers is required')
    selected = tuple(tickers)
    if (len(selected) not in (2, len(TICKERS)) or any(not isinstance(t, str) or t not in TICKERS for t in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('Tickers must be distinct, eligible, and form a pair or the full universe')
    return selected


def _text(value, name):
    if not isinstance(value, str) or not value.strip(): raise ValueError(name+' requires actual nonempty text')
    return value


def _ids(value, name):
    if (not isinstance(value, list) or any(not isinstance(v, str) or not v or v != v.strip() for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(name+' requires distinct actual evidence identifiers')
    return tuple(value)


def short_reference(original, *, kind='e'):
    if not isinstance(original, str) or not original or original != original.strip() or kind not in ('e', 'm'):
        raise ValueError('An actual evidence or memory identifier is required')
    return kind+hashlib.sha256(original.encode('utf-8')).hexdigest()[:12]


def _reference_map(identifiers, *, kind):
    result = {}
    for original in identifiers:
        short = short_reference(original, kind=kind)
        if short in result and result[short] != original: raise ValueError('Short evidence reference collision')
        result[short] = original
    return dict(sorted(result.items()))


def _merge_references(target, addition):
    for short, original in addition.items():
        if short in target and target[short] != original: raise ValueError('Short evidence reference collision')
        target[short] = original


def _company_projection(document, tickers):
    if not isinstance(document, Mapping) or not isinstance(document.get('companies'), Mapping):
        raise ValueError('A real company context document is required')
    selected = _selected(tickers); companies = document['companies']
    if any(t not in companies for t in selected): raise ValueError('Requested company context is missing')
    source_raw = document.get('as_of', document.get('archive', {}).get('as_of'))
    source = aware(source_raw)
    references, output = {}, {}
    for ticker in selected:
        company = companies[ticker]
        if not isinstance(company, Mapping): raise ValueError('A real company context object is required')
        source_ids = _ids(company.get('source_ids', []), 'Source evidence')
        available = set(source_ids)
        used = []
        def refs(value):
            identifiers = _ids(value, 'Cited evidence')
            if not set(identifiers) <= available: raise ValueError('A cited source is absent from the actual company evidence')
            used.extend(identifiers)
            return [short_reference(value) for value in identifiers]
        result = {'recent_operations_summary': _text(company.get('recent_operations_summary'), 'Company summary')}
        if 'summary_evidence_uuids' in company:
            result['summary_evidence_refs'] = refs(company['summary_evidence_uuids'])
        for field in ('company_facts', 'analyst_outlook'):
            if field not in company: continue
            if not isinstance(company[field], list): raise ValueError('Company claims must be actual ordered records')
            claims = []
            for claim in company[field]:
                if not isinstance(claim, Mapping) or set(claim) != {'statement', 'evidence_uuids'}:
                    raise ValueError('Company facts and outlook need their actual statement and source evidence')
                claims.append({'statement': _text(claim['statement'], 'Company claim'),
                    'evidence_refs': refs(claim['evidence_uuids'])})
            result[field] = claims
        if 'limitations' in company: result['limitations'] = _text(company['limitations'], 'Company limitations')
        if 'source_date_bounds' in company:
            bounds = company['source_date_bounds']
            if not isinstance(bounds, Mapping) or set(bounds) != {'first', 'last'}:
                raise ValueError('Actual first/last source publication bounds required')
            first, last = aware(bounds['first']), aware(bounds['last'])
            if first > last or last > source: raise ValueError('Company evidence cannot exceed the source cutoff')
            result['source_date_bounds'] = {'first': first.isoformat(), 'last': last.isoformat()}
        _merge_references(references, _reference_map(used, kind='e'))
        output[ticker] = result
    compact = {'as_of': source.isoformat(),
        'source_basis': document.get('content_basis', 'partial archived news evidence'), 'companies': output}
    if not isinstance(compact['source_basis'], str): raise ValueError('A textual source basis is required')
    return compact, dict(sorted(references.items()))


def compact_company_context(document, *, tickers=None):
    """Full statements, selected companies, and necessary short references only."""
    return _company_projection(document, tickers)[0]


def company_reference_map(document, *, tickers=None):
    """Private-only short->original mapping for references present in the view."""
    return _company_projection(document, tickers)[1]


def _hit(hit):
    value = hit.to_prompt() if callable(getattr(hit, 'to_prompt', None)) else hit
    if not isinstance(value, Mapping): raise ValueError('An actual retrieved memory unit is required')
    _text(value.get('unit_id'), 'Memory identifier')
    return value


def compact_reflection_hit(hit, *, reflection_text=None):
    """Keep the bank ID and whole reflection, without rank/audit/source payloads.

    The caller may supply its complete news-free historical display rendering.
    This projection never abbreviates text or changes the actual bank identity.
    """
    value = _hit(hit)
    text = value.get('reflection') if reflection_text is None else reflection_text
    return {'unit_id': value['unit_id'], 'reflection': _text(text, 'Complete reflection')}


def compact_reflection_hits(hits, *, reflection_renderer=None):
    """Preserve accepted retrieval order and optional pure display rendering."""
    values = [_hit(hit) for hit in hits]
    if len({row['unit_id'] for row in values}) != len(values):
        raise ValueError('A ranked prompt cannot repeat one memory unit')
    return [compact_reflection_hit(row, reflection_text=reflection_renderer(row) if reflection_renderer else None)
        for row in values]


_PUBLISHED = re.compile(r'\APublished (?P<at>\d{4}-\d{2}-\d{2}T[0-9:.]+(?:Z|[+-]\d{2}:\d{2})) '
    r'\[(?P<identity>[^\]\r\n]+)\]: (?P<summary>[\s\S]+)\Z')
_NEWS_FIELDS = frozenset({'news', 'news_a', 'news_b', 'news_A_summary', 'news_B_summary', 'past_week_news_summary'})
_ISO_TIME = r'\d{4}-\d{2}-\d{2}T[0-9:.]+(?:Z|[+-]\d{2}:\d{2})'
_NATIVE_NEWS = re.compile(r'\A(?P<prefix>[ \t]*(?:\d+\.[ \t]+)?\[(?:'
    +'|'.join(re.escape(t) for t in TICKERS)+r')\][ \t]+(?:(?:Published)?'+_ISO_TIME+r':[ \t]+)?Published )'
    r'(?P<at>'+_ISO_TIME+r') \[(?P<identity>[^\]\r\n]+)\]: (?P<summary>[^\r\n]+)(?P<ending>\r?\n)?\Z')


def _published(text):
    if not isinstance(text, str): raise ValueError('Actual news text is required')
    match = _PUBLISHED.fullmatch(text)
    if match is None: return None
    aware(match['at'])
    short_reference(match['identity'])
    return match


def compact_news_text(text):
    """Only shorten the exact data.observation article header; keep its body."""
    match = _published(text)
    if match is None: return text
    return 'Published '+match['at']+' ['+short_reference(match['identity'])+']: '+match['summary']


def _news_tree(value):
    if isinstance(value, str):
        compact = compact_news_text(value)
        if compact != value: return compact
        # Some native news fields contain a JSON-encoded article list. Decode
        # only that named news field, preserving every decoded source string.
        try: nested = json.loads(value)
        except (ValueError, TypeError): return value
        if not isinstance(nested, (Mapping, list)): return value
        projected = _news_tree(nested)
        return json.dumps(projected, ensure_ascii=False) if projected != nested else value
    if isinstance(value, list): return [_news_tree(row) for row in value]
    if isinstance(value, Mapping): return {key: _news_tree(row) for key, row in value.items()}
    return copy.deepcopy(value)


def compact_payload_news(value):
    """Visit named news fields, never rewrite reflection/reason/ID strings."""
    if isinstance(value, Mapping):
        return {key: _news_tree(row) if key in _NEWS_FIELDS else compact_payload_news(row)
            if isinstance(row, (Mapping, list)) else copy.deepcopy(row) for key, row in value.items()}
    if isinstance(value, list): return [compact_payload_news(row) for row in value]
    return copy.deepcopy(value)


def _native_news_line(line):
    match = _NATIVE_NEWS.fullmatch(line)
    if match is not None:
        aware(match['at']); short_reference(match['identity'])
    return match


def compact_native_prompt(text):
    """Shorten only numbered/ticker-labelled native article headers in place.

    Memory row numbers, ticker prefixes, all dates, whitespace and article text
    remain byte-for-byte identical. Reflection prose and bare IDs are untouched.
    """
    if not isinstance(text, str): raise ValueError('The actual native prompt text is required')
    result = []
    for line in text.splitlines(keepends=True):
        match = _native_news_line(line)
        if match is None: result.append(line)
        else:
            start, end = match.span('identity')
            result.append(line[:start]+short_reference(match['identity'])+line[end:])
    return ''.join(result)


def native_prompt_reference_map(text):
    """Private lookup for exact news headers present inside a native prompt."""
    if not isinstance(text, str): raise ValueError('The actual native prompt text is required')
    references = {}
    for line in text.splitlines(keepends=True):
        match = _native_news_line(line)
        if match is not None: _merge_references(references, _reference_map([match['identity']], kind='e'))
    return dict(sorted(references.items()))


def _limitations(coverage, ticker):
    if isinstance(coverage, Mapping):
        row = coverage.get(ticker, coverage if 'status' in coverage else None)
    else: row = {'status': coverage} if isinstance(coverage, str) else None
    if not isinstance(row, Mapping): return ['Coverage information unavailable.']
    result = []
    status = row.get('status')
    if status == 'unavailable': result.append('News unavailable.')
    elif status != 'complete': result.append('Coverage is partial.' if status == 'partial' else 'Coverage information unavailable.')
    if row.get('live_status') == 'unavailable': result.append('Live feed unavailable.')
    if row.get('live_capped') is True: result.append('Source search capped.')
    if row.get('window_truncated') is True: result.append('Publication window may be truncated.')
    if row.get('content_basis') in ('partial_headline', 'headline_only_partial'): result.append('Some evidence is headline-only.')
    for field, suffix in (('invalid_publication_count', 'items lack a valid publication timestamp.'),
                          ('conflict_count', 'conflicting publication groups omitted.')):
        count = row.get(field, 0)
        if type(count) is not int or count < 0: raise ValueError('Actual nonnegative news coverage counts required')
        if count: result.append(str(count)+' '+suffix)
    return result


def _observation_projection(observation, tickers):
    if not isinstance(observation, Mapping): raise ValueError('An actual observation object is required')
    selected = _selected(tickers); at = aware(observation['timestamp'])
    prices, news = observation.get('prices'), observation.get('news')
    if not isinstance(prices, Mapping) or not isinstance(news, Mapping): raise ValueError('Released prices and news are required')
    result = {'timestamp': observation['timestamp'], 'prices': {}, 'news': {}}; references = {}; limitations = {}
    for ticker in selected:
        if ticker not in prices or ticker not in news: raise ValueError('Requested ticker observation is missing')
        price = prices[ticker]
        if type(price) not in (int, float) or not math.isfinite(price) or price <= 0: raise ValueError('Released finite positive prices required')
        if not isinstance(news[ticker], list): raise ValueError('Actual ordered news rows required')
        result['prices'][ticker] = price; result['news'][ticker] = []
        missing_time = False
        for text in news[ticker]:
            match = _published(text)
            if match is not None:
                if aware(match['at']) >= at: raise ValueError('Only news published before this release may enter its prompt')
                _merge_references(references, _reference_map([match['identity']], kind='e'))
            else: missing_time = True
            result['news'][ticker].append(compact_news_text(text))
        notes = _limitations(observation.get('news_coverage'), ticker)
        if missing_time: notes.append('Some news lacks an exact publication timestamp.')
        if notes: limitations[ticker] = notes
    if limitations: result['news_limitations'] = limitations
    return result, dict(sorted(references.items()))


def compact_observation(observation, tickers=None):
    """Scope the point-in-time model view without mutating the released input."""
    return _observation_projection(observation, tickers)[0]


def observation_reference_map(observation, tickers=None):
    """Private article lookup for the references actually present in this view."""
    return _observation_projection(observation, tickers)[1]
