import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from django import db
from django.core.management import BaseCommand

from translations.models.language import Language
from translations.models.textbase import TextBase
from translations.models.translation import Translation
from config import TRANSLATE_API_KEY

# Brand names and fixed strings that must survive translation unchanged
_PROTECTED_BRANDS = ['BackgroundRemover']
# Regex to match emoji characters (broad Unicode emoji ranges)
_EMOJI_RE = re.compile(
    '[\U0001F600-\U0001F64F'   # emoticons
    '\U0001F300-\U0001F5FF'    # symbols & pictographs
    '\U0001F680-\U0001F6FF'    # transport & map
    '\U0001F1E0-\U0001F1FF'    # flags
    '\U00002702-\U000027B0'    # dingbats
    '\U0000FE00-\U0000FE0F'    # variation selectors
    '\U0001F900-\U0001F9FF'    # supplemental symbols
    '\U0001FA00-\U0001FA6F'    # chess symbols
    '\U0001FA70-\U0001FAFF'    # symbols extended-A
    '\U00002600-\U000026FF'    # misc symbols
    '\U0000200D'               # zero width joiner
    '\U00002B50\U00002B55'     # star, circle
    '\U0000231A-\U0000231B'    # watch, hourglass
    '\U000023E9-\U000023F3'    # media controls
    '\U000023F8-\U000023FA'    # more media
    ']+', re.UNICODE
)

API_BASE = 'https://api.translateapi.ai/api/v1'
MAX_TEXTS = 100  # API max texts per batch
MAX_ITEMS = 300  # API max total items (texts x languages)
POLL_INTERVAL = 5  # Poll every 3-5s (API recommendation)
MAX_WORKERS = 2  # Max concurrent jobs (API hard limit: 2)
MAX_POLL_TIME = 2700  # 45 min job timeout (API limit)
SYNC_LANGS_PER_CALL = 20  # Langs per sync API call (max 50, 20 is safe/fast)

# Language code remapping (our DB code -> API code)
LANG_REMAP = {'jv': 'jw'}

# Translation validation patterns
_TEMPLATE_TAG_CHARS = ('{%', '%}', '{{', '}}')
_ERROR_PREFIXES = ('[Error:', '[error:', 'Error:')
_REJECT_SUBSTRINGS = ('CUDA', 'out of memory', 'translation engines failed', 'MADLAD batch',
                      'traceback', 'RuntimeError', 'torch.cuda')
SKIP_LANGS = {
    'as', 'be', 'ca', 'cy', 'eo', 'eu', 'ga', 'gl', 'ht', 'jv',
    'ku', 'lb', 'mi', 'ps', 'sd', 'su', 'tt', 'ug', 'yi',
}


class Command(BaseCommand):
    help = 'Translate all untranslated TextBase entries using translateapi.ai batch API'

    def add_arguments(self, parser):
        parser.add_argument('--lang', type=str, help='Only translate to a specific language')
        parser.add_argument('--batch-size', type=int, default=0, help='Texts per API call (auto-calculated if 0)')
        parser.add_argument('--dry-run', action='store_true', help='Show what would be translated')
        parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Concurrent batch workers')
        parser.add_argument('--langs-per-call', type=int, default=0, help='Target languages per API call (auto-calculated if 0)')
        parser.add_argument('--retry-failed', type=str, help='Retry specific failed languages (comma-separated, e.g. "my,km,hy")')
        parser.add_argument('--retry-all-fallbacks', action='store_true', help='Retry all languages that have English fallback translations')
        parser.add_argument('--fill-missing', action='store_true', help='Translate keys that have no Translation record at all for each language')
        parser.add_argument('--sync', action='store_true', help='Use sync API instead of batch (slower but works when batch workers are down)')

    @staticmethod
    def _is_valid_translation(text, source_text=''):
        """Validate translation output. Returns True if text looks like a valid translation."""
        if not text or not isinstance(text, str):
            return False
        text = text.strip()
        if not text:
            return False
        # Reject error messages
        for prefix in _ERROR_PREFIXES:
            if text.startswith(prefix):
                return False
        # Reject text containing Django template tags
        for tag in _TEMPLATE_TAG_CHARS:
            if tag in text:
                return False
        # Reject text containing error/debug substrings
        text_lower = text.lower()
        for sub in _REJECT_SUBSTRINGS:
            if sub.lower() in text_lower:
                # Allow if the source text also contains this substring (legitimate content)
                if source_text and sub.lower() in source_text.lower():
                    continue
                return False
        # Reject repetitive garbage (single char >40% of text, text >50 chars)
        if len(text) > 50:
            for char in set(text):
                if char == ' ':
                    continue
                if text.count(char) / len(text) > 0.4 and text.count(char) > 20:
                    return False
        # Reject if translation is >5x longer than source (likely garbage)
        if source_text and len(text) > max(len(source_text) * 5, 500):
            return False
        # Reject if source contained protected brands but translation lost them
        for brand in _PROTECTED_BRANDS:
            if brand in source_text and brand not in text:
                return False
        return True

    @staticmethod
    def _protect_variables(text):
        """Replace brand names and emojis with numeric tokens before translation.
        Uses distinctive numbers (88800, 88801, ...) that translation APIs preserve
        reliably, unlike bracket-based placeholders which get mangled.
        Returns (protected_text, mapping) where mapping is needed to restore later."""
        mapping = {}
        result = text
        token_idx = 0
        # Protect brand names with numeric tokens
        for brand in _PROTECTED_BRANDS:
            token = f'88{token_idx:03d}'
            if brand in result:
                mapping[token] = brand
                result = result.replace(brand, token)
                token_idx += 1
        # Protect emoji sequences with numeric tokens
        for match in _EMOJI_RE.finditer(result):
            emoji = match.group()
            token = f'88{token_idx:03d}'
            mapping[token] = emoji
            token_idx += 1
        # Replace emojis in one pass (reverse order to keep positions valid)
        for token, emoji in sorted(mapping.items(), reverse=True):
            if not any(emoji == b for b in _PROTECTED_BRANDS):
                result = result.replace(emoji, token, 1)
        return result, mapping

    @staticmethod
    def _restore_variables(text, mapping):
        """Restore brand names and emojis from numeric tokens after translation.
        Also cleans up any remnants of old [[BRAND0]]-style placeholders."""
        if not mapping:
            # Still clean up any legacy placeholders that might exist
            text = re.sub(r'\[+[A-Z]*BRAND\d*\]*', '', text)
            return text
        result = text
        for token, original in mapping.items():
            result = result.replace(token, original)
        # Clean up any legacy [[BRAND0]] placeholders from old runs
        result = re.sub(r'\[+[A-Z]*BRAND\d*\]*', '', result)
        return result

    def _calc_batch_params(self, num_langs, user_batch_size, user_langs_per_call):
        """Calculate batch_size and langs_per_call respecting API limits.
        API limits: max 100 texts, max 300 total items (texts x languages).
        Best practice: 1 target language per batch, 50-100 texts."""
        langs_per_call = min(user_langs_per_call or 1, 50, num_langs)
        batch_size = min(user_batch_size or MAX_TEXTS, MAX_TEXTS, MAX_ITEMS // langs_per_call)
        # Ensure we don't exceed MAX_ITEMS
        while batch_size * langs_per_call > MAX_ITEMS:
            if langs_per_call > 1:
                langs_per_call -= 1
            else:
                batch_size = MAX_ITEMS
                break
        return batch_size, langs_per_call

    def handle(self, *args, **options):
        self._use_sync = options.get('sync', False)
        if self._use_sync:
            self.stdout.write('Using SYNC API (slower, use when batch workers are down)')

        retry_failed = options.get('retry_failed')
        if retry_failed:
            return self._handle_retry(retry_failed, options)

        if options.get('retry_all_fallbacks'):
            return self._handle_retry_all(options)

        if options.get('fill_missing'):
            return self._handle_fill_missing(options)

        target_lang = options.get('lang')
        dry_run = options.get('dry_run', False)
        max_workers = min(options.get('workers', MAX_WORKERS), MAX_WORKERS)
        if options.get('workers', MAX_WORKERS) > MAX_WORKERS:
            self.stdout.write(self.style.WARNING(
                f'Workers capped at {MAX_WORKERS} (API limit: max 2 concurrent jobs)'
            ))

        variables = TextBase.objects.filter(translated=False)
        if not variables.exists():
            self.stdout.write('Nothing to translate')
            return

        if target_lang:
            langs = list(Language.objects.filter(iso=target_lang))
            if not langs:
                self.stdout.write(self.style.ERROR(f'Language "{target_lang}" not found'))
                return
        else:
            langs = list(Language.objects.exclude(iso='en'))

        items = list(variables)

        # Build lang codes with remapping: [(api_code, db_code), ...]
        lang_list = []
        for lang in langs:
            if lang.iso in SKIP_LANGS:
                continue
            api_code = LANG_REMAP.get(lang.iso, lang.iso)
            lang_list.append((api_code, lang.iso))

        # Calculate optimal batch params respecting API limits
        if self._use_sync:
            # Sync mode: 1 text per API call, all langs at once (chunked by SYNC_LANGS_PER_CALL internally)
            batch_size = options.get('batch_size') or 50  # DB commit batch
            langs_per_call = len(lang_list)  # All langs per text
            lang_chunks = [lang_list]
            sync_calls_per_text = (len(lang_list) + SYNC_LANGS_PER_CALL - 1) // SYNC_LANGS_PER_CALL
        else:
            batch_size, langs_per_call = self._calc_batch_params(
                len(lang_list),
                options.get('batch_size') or 0,
                options.get('langs_per_call') or 0,
            )
            # Split languages into chunks
            lang_chunks = []
            for i in range(0, len(lang_list), langs_per_call):
                lang_chunks.append(lang_list[i:i + langs_per_call])

        items_per_call = batch_size * langs_per_call
        total_text_batches = (len(items) + batch_size - 1) // batch_size

        total_api_calls = total_text_batches * len(lang_chunks)
        self.stdout.write(f'Translating {len(items)} entries into {len(lang_list)} languages')
        if self._use_sync:
            self.stdout.write(f'Sync mode: {batch_size} texts/commit, {SYNC_LANGS_PER_CALL} langs/call, {sync_calls_per_text} calls/text')
        else:
            self.stdout.write(f'Batch: {batch_size} texts x {langs_per_call} langs = {items_per_call} items/call (API max: {MAX_ITEMS})')
            self.stdout.write(f'{total_text_batches} text batches x {len(lang_chunks)} lang chunks = {total_api_calls} API calls')
        self.stdout.write(f'Running {max_workers} concurrent API calls')

        if dry_run:
            self.stdout.write('\nDry run — no translations will be saved')
            return

        failed_langs = {}  # api_code -> consecutive fail count
        total_saved = 0
        start_time = time.time()

        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start:batch_start + batch_size]
            protected = [self._protect_variables(item.text) for item in batch]
            text_strings = [p[0] for p in protected]
            var_mappings = [p[1] for p in protected]
            batch_num = batch_start // batch_size + 1
            batch_start_time = time.time()

            self.stdout.write(
                f'\n{"="*60}\n'
                f'Batch {batch_num}/{total_text_batches}: {len(batch)} texts '
                f'({batch[0].code_name} ... {batch[-1].code_name})'
            )

            # Process lang chunks concurrently
            batch_saved = 0
            batch_failed = 0

            def translate_multi_lang(lang_chunk):
                """Translate texts to multiple languages in one API call."""
                active = [(ac, dc) for ac, dc in lang_chunk if failed_langs.get(ac, 0) < 3]
                if not active:
                    return (lang_chunk, None, True)
                active_api_codes = [ac for ac, dc in active]
                translations = self._translate_batch_multi(text_strings, active_api_codes)
                return (lang_chunk, translations, False)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(translate_multi_lang, chunk): chunk
                    for chunk in lang_chunks
                }

                completed = 0
                for future in as_completed(futures):
                    lang_chunk, translations, all_skipped = future.result()
                    completed += 1

                    if translations is not None:
                        db.close_old_connections()
                        active_pairs = [(ac, dc) for ac, dc in lang_chunk if failed_langs.get(ac, 0) < 3]
                        for api_code, db_code in active_pairs:
                            failed_langs.pop(api_code, None)
                            for i, item in enumerate(batch):
                                if i < len(translations) and isinstance(translations[i], dict):
                                    text = translations[i].get(api_code)
                                    if text is None:
                                        continue  # skip missing, don't save English fallback
                                    text = str(text).replace('&#39;', "'").replace('&quot;', '"')
                                    text = self._restore_variables(text, var_mappings[i])
                                else:
                                    continue  # skip missing results
                                if not self._is_valid_translation(text, item.text):
                                    continue
                                Translation.register_text_translated({
                                    'language': db_code,
                                    'code_name': item.code_name,
                                    'text': text,
                                })
                            batch_saved += 1

                        chunk_langs = ', '.join(dc for _, dc in active_pairs[:5])
                        if len(active_pairs) > 5:
                            chunk_langs += f' +{len(active_pairs)-5} more'
                        self.stdout.write(f'  [{completed}/{len(lang_chunks)}] OK: {chunk_langs}')
                    elif not all_skipped:
                        # API failure — skip, don't save English fallback
                        active_pairs = [(ac, dc) for ac, dc in lang_chunk if failed_langs.get(ac, 0) < 3]
                        for api_code, db_code in active_pairs:
                            failed_langs[api_code] = failed_langs.get(api_code, 0) + 1
                            batch_failed += 1
                        chunk_langs = ', '.join(dc for _, dc in active_pairs[:5])
                        self.stdout.write(self.style.WARNING(
                            f'  [{completed}/{len(lang_chunks)}] FAILED: {chunk_langs} — skipped'
                        ))

            # Mark batch as translated
            db.close_old_connections()
            TextBase.objects.filter(
                pk__in=[item.pk for item in batch]
            ).update(translated=True)

            total_saved += batch_saved
            batch_elapsed = time.time() - batch_start_time
            total_elapsed = time.time() - start_time
            remaining_batches = total_text_batches - batch_num
            eta = (total_elapsed / batch_num) * remaining_batches

            self.stdout.write(self.style.SUCCESS(
                f'  Batch {batch_num} done in {batch_elapsed:.1f}s: '
                f'{batch_saved} langs OK, {batch_failed} failed'
            ))
            if remaining_batches > 0:
                self.stdout.write(f'  ETA: ~{eta/60:.1f} min remaining ({remaining_batches} batches left)')

        total_elapsed = time.time() - start_time
        permanently_failed = {k for k, v in failed_langs.items() if v >= 3}
        self.stdout.write(self.style.SUCCESS(
            f'\nComplete! {total_saved} language-batches saved in {total_elapsed/60:.1f} min. '
            f'{len(permanently_failed)} languages permanently failed.'
        ))
        if permanently_failed:
            self.stdout.write(f'Permanently failed (3+ consecutive): {", ".join(sorted(permanently_failed))}')

    def _handle_retry(self, retry_langs_str, options):
        """Retry translation for specific languages that previously got English fallback."""
        dry_run = options.get('dry_run', False)

        retry_isos = [l.strip() for l in retry_langs_str.split(',') if l.strip()]
        self.stdout.write(f'Retrying failed languages: {", ".join(retry_isos)}')

        all_items = list(TextBase.objects.all())
        if not all_items:
            self.stdout.write('No TextBase entries found')
            return

        for db_code in retry_isos:
            api_code = LANG_REMAP.get(db_code, db_code)
            english_map = {item.code_name: item.text for item in all_items}
            existing = Translation.objects.filter(language=db_code)
            fallback_codes = []
            for t in existing:
                if t.code_name in english_map and t.text == english_map[t.code_name]:
                    fallback_codes.append(t.code_name)

            if not fallback_codes:
                self.stdout.write(f'  [{db_code}] No fallback entries found — skipping')
                continue

            self.stdout.write(f'  [{db_code}] Found {len(fallback_codes)} fallback entries to retry')
            if dry_run:
                continue

            items_to_retry = [item for item in all_items if item.code_name in set(fallback_codes)]
            batch_size = min(MAX_TEXTS, MAX_ITEMS)  # single lang
            total_batches = (len(items_to_retry) + batch_size - 1) // batch_size
            saved = 0

            for batch_start in range(0, len(items_to_retry), batch_size):
                batch = items_to_retry[batch_start:batch_start + batch_size]
                protected = [self._protect_variables(item.text) for item in batch]
                text_strings = [p[0] for p in protected]
                var_mappings = [p[1] for p in protected]
                batch_num = batch_start // batch_size + 1

                self.stdout.write(f'  [{db_code}] Batch {batch_num}/{total_batches} ({len(batch)} texts)')
                translations = self._translate_batch_multi(text_strings, [api_code])

                if translations is not None:
                    db.close_old_connections()
                    for i, item in enumerate(batch):
                        if i < len(translations) and isinstance(translations[i], dict):
                            text = translations[i].get(api_code)
                            if text is None:
                                continue
                            text = str(text).replace('&#39;', "'").replace('&quot;', '"')
                            text = self._restore_variables(text, var_mappings[i])
                        else:
                            continue
                        if not self._is_valid_translation(text, item.text):
                            continue
                        Translation.register_text_translated({
                            'language': db_code,
                            'code_name': item.code_name,
                            'text': text,
                        })
                        saved += 1
                    self.stdout.write(self.style.SUCCESS(f'  [{db_code}] Batch {batch_num} OK'))
                else:
                    self.stdout.write(self.style.ERROR(f'  [{db_code}] Batch {batch_num} FAILED'))

            self.stdout.write(self.style.SUCCESS(f'  [{db_code}] {saved} entries re-translated'))

        self.stdout.write(self.style.SUCCESS('Retry complete!'))

    def _handle_fill_missing(self, options):
        """Find and translate keys that have NO Translation record for each language."""
        dry_run = options.get('dry_run', False)
        max_workers = min(options.get('workers', MAX_WORKERS), MAX_WORKERS)
        target_lang = options.get('lang')

        all_items = list(TextBase.objects.all())
        if not all_items:
            self.stdout.write('No TextBase entries found')
            return

        all_code_names = {item.code_name for item in all_items}

        if target_lang:
            langs = list(Language.objects.filter(iso=target_lang))
        else:
            langs = list(Language.objects.exclude(iso='en'))

        # Find missing translations per language
        langs_to_fill = {}
        for lang in langs:
            existing_codes = set(
                Translation.objects.filter(language=lang.iso)
                .values_list('code_name', flat=True)
            )
            missing_codes = all_code_names - existing_codes
            if missing_codes:
                langs_to_fill[lang.iso] = sorted(missing_codes)

        if not langs_to_fill:
            self.stdout.write('No missing translations found — all languages have all keys')
            return

        total_missing = sum(len(v) for v in langs_to_fill.values())
        self.stdout.write(f'Found {total_missing} missing translations across {len(langs_to_fill)} languages')
        self.stdout.write(f'Processing {max_workers} languages in parallel')

        if dry_run:
            for iso, codes in sorted(langs_to_fill.items()):
                self.stdout.write(f'  {iso}: {len(codes)} missing')
            return

        start_time = time.time()
        total_saved = 0
        total_failed = 0
        items_by_code = {item.code_name: item for item in all_items}

        def fill_one_lang(db_code, missing_codes):
            """Translate all missing keys for a single language."""
            api_code = LANG_REMAP.get(db_code, db_code)
            items_to_translate = [items_by_code[cn] for cn in missing_codes if cn in items_by_code]
            batch_size = min(MAX_TEXTS, MAX_ITEMS)
            saved = 0

            for batch_start in range(0, len(items_to_translate), batch_size):
                batch = items_to_translate[batch_start:batch_start + batch_size]
                protected = [self._protect_variables(item.text) for item in batch]
                text_strings = [p[0] for p in protected]
                var_mappings = [p[1] for p in protected]
                translations = self._translate_batch_multi(text_strings, [api_code])

                if translations is not None:
                    db.close_old_connections()
                    for i, item in enumerate(batch):
                        if i < len(translations) and isinstance(translations[i], dict):
                            text = translations[i].get(api_code)
                            if text is None:
                                continue
                            text = str(text).replace('&#39;', "'").replace('&quot;', '"')
                            text = self._restore_variables(text, var_mappings[i])
                        else:
                            continue
                        if not self._is_valid_translation(text, item.text):
                            continue
                        Translation.register_text_translated({
                            'language': db_code,
                            'code_name': item.code_name,
                            'text': text,
                        })
                        saved += 1
            return (db_code, saved, len(missing_codes))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fill_one_lang, db_code, codes): db_code
                for db_code, codes in sorted(langs_to_fill.items())
            }

            completed = 0
            for future in as_completed(futures):
                db_code, saved, total_codes = future.result()
                completed += 1
                total_saved += saved
                if saved < total_codes:
                    total_failed += (total_codes - saved)
                elapsed = time.time() - start_time
                remaining = len(langs_to_fill) - completed
                eta = (elapsed / completed) * remaining if completed > 0 else 0
                self.stdout.write(self.style.SUCCESS(
                    f'  [{completed}/{len(langs_to_fill)}] {db_code}: {saved}/{total_codes} translated '
                    f'({elapsed:.0f}s elapsed, ~{eta/60:.1f}min remaining)'
                ))

        total_elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f'\nFill complete! {total_saved} translations created in {total_elapsed/60:.1f} min. '
            f'{total_failed} still missing.'
        ))

    def _handle_retry_all(self, options):
        """Retry all languages that have English fallback translations, in parallel."""
        dry_run = options.get('dry_run', False)
        max_workers = min(options.get('workers', MAX_WORKERS), MAX_WORKERS)

        all_items = list(TextBase.objects.all())
        if not all_items:
            self.stdout.write('No TextBase entries found')
            return

        english_map = {item.code_name: item.text for item in all_items}
        langs = list(Language.objects.exclude(iso='en'))

        # Find all languages with fallbacks
        langs_to_retry = {}
        for lang in langs:
            existing = Translation.objects.filter(language=lang.iso)
            fallback_codes = [t.code_name for t in existing
                              if t.code_name in english_map and t.text == english_map[t.code_name]]
            if fallback_codes:
                langs_to_retry[lang.iso] = fallback_codes

        if not langs_to_retry:
            self.stdout.write('No fallback translations found')
            return

        total_fallbacks = sum(len(v) for v in langs_to_retry.values())
        self.stdout.write(f'Found {total_fallbacks} fallback translations across {len(langs_to_retry)} languages')
        self.stdout.write(f'Processing {max_workers} languages in parallel')

        if dry_run:
            for iso, codes in sorted(langs_to_retry.items()):
                self.stdout.write(f'  {iso}: {len(codes)} fallbacks')
            return

        start_time = time.time()
        total_saved = 0
        total_failed = 0

        def retry_one_lang(db_code, fallback_codes):
            """Retry all fallback translations for a single language."""
            api_code = LANG_REMAP.get(db_code, db_code)
            items_to_retry = [item for item in all_items if item.code_name in set(fallback_codes)]
            batch_size = min(MAX_TEXTS, MAX_ITEMS)
            saved = 0

            for batch_start in range(0, len(items_to_retry), batch_size):
                batch = items_to_retry[batch_start:batch_start + batch_size]
                protected = [self._protect_variables(item.text) for item in batch]
                text_strings = [p[0] for p in protected]
                var_mappings = [p[1] for p in protected]
                translations = self._translate_batch_multi(text_strings, [api_code])

                if translations is not None:
                    db.close_old_connections()
                    for i, item in enumerate(batch):
                        if i < len(translations) and isinstance(translations[i], dict):
                            text = translations[i].get(api_code)
                            if text is None:
                                continue
                            text = str(text).replace('&#39;', "'").replace('&quot;', '"')
                            text = self._restore_variables(text, var_mappings[i])
                        else:
                            continue
                        if not self._is_valid_translation(text, item.text):
                            continue
                        Translation.register_text_translated({
                            'language': db_code,
                            'code_name': item.code_name,
                            'text': text,
                        })
                        saved += 1
            return (db_code, saved, len(fallback_codes))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(retry_one_lang, db_code, codes): db_code
                for db_code, codes in sorted(langs_to_retry.items())
            }

            completed = 0
            for future in as_completed(futures):
                db_code, saved, total_codes = future.result()
                completed += 1
                total_saved += saved
                if saved < total_codes:
                    total_failed += (total_codes - saved)
                elapsed = time.time() - start_time
                remaining = len(langs_to_retry) - completed
                eta = (elapsed / completed) * remaining if completed > 0 else 0
                self.stdout.write(self.style.SUCCESS(
                    f'  [{completed}/{len(langs_to_retry)}] {db_code}: {saved}/{total_codes} retranslated '
                    f'({elapsed:.0f}s elapsed, ~{eta/60:.1f}min remaining)'
                ))

        total_elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f'\nRetry complete! {total_saved} translations fixed in {total_elapsed/60:.1f} min. '
            f'{total_failed} still failed.'
        ))

    def _translate_sync_one(self, text, target_languages, _retry=0):
        """Translate a single text to multiple languages via sync endpoint.
        Returns dict {lang_code: translated_text} or None."""
        headers = {
            'Authorization': f'Bearer {TRANSLATE_API_KEY}',
            'Content-Type': 'application/json',
        }
        payload = {'text': text, 'source_language': 'en'}
        if len(target_languages) == 1:
            payload['target_language'] = target_languages[0]
        else:
            payload['target_languages'] = target_languages

        try:
            r = requests.post(f'{API_BASE}/translate/', headers=headers, json=payload, timeout=300)
        except requests.exceptions.RequestException as e:
            if _retry < 2:
                time.sleep(10)
                return self._translate_sync_one(text, target_languages, _retry=_retry + 1)
            return None

        if r.status_code == 200:
            data = r.json()
            translations = data.get('translations', {})
            if isinstance(translations, dict):
                return translations
            if 'translated_text' in data and len(target_languages) == 1:
                return {target_languages[0]: data['translated_text']}
            return None
        elif r.status_code == 429:
            time.sleep(30)
            return self._translate_sync_one(text, target_languages, _retry=_retry)
        elif r.status_code == 402:
            return None
        else:
            if _retry < 2:
                time.sleep(10)
                return self._translate_sync_one(text, target_languages, _retry=_retry + 1)
            return None

    def _translate_sync_multi(self, texts, target_languages):
        """Translate multiple texts via sync endpoint (one at a time).
        Returns list of dicts [{lang: text}, ...] matching batch format."""
        # Split languages into chunks of SYNC_LANGS_PER_CALL
        lang_chunks = []
        for i in range(0, len(target_languages), SYNC_LANGS_PER_CALL):
            lang_chunks.append(target_languages[i:i + SYNC_LANGS_PER_CALL])

        results = []
        for idx, text in enumerate(texts):
            merged = {}
            for chunk in lang_chunks:
                trans = self._translate_sync_one(text, chunk)
                if trans:
                    merged.update(trans)
            results.append(merged)
            if (idx + 1) % 5 == 0 or idx == len(texts) - 1:
                ok = sum(1 for r in results if r)
                self.stdout.write(f'    sync {idx+1}/{len(texts)}: {ok} OK, {len(merged)} langs')
        return results if any(r for r in results) else None

    def _translate_batch_multi(self, texts, target_languages, _retry=0):
        """Submit multi-language translation job. Returns list of dicts [{lang: text}, ...] or None.
        Uses sync endpoint when --sync flag is set, batch endpoint otherwise.
        Retries up to 3 times on transient errors (service restarts, stalls)."""
        if getattr(self, '_use_sync', False):
            return self._translate_sync_multi(texts, target_languages)

        headers = {
            'Authorization': f'Bearer {TRANSLATE_API_KEY}',
            'Content-Type': 'application/json',
        }

        payload = {
            'texts': texts,
            'source_language': 'en',
        }
        if len(target_languages) == 1:
            payload['target_language'] = target_languages[0]
        else:
            payload['target_languages'] = target_languages

        lang_str = ','.join(target_languages[:3])
        if len(target_languages) > 3:
            lang_str += f'+{len(target_languages)-3}'

        # Submit batch job
        try:
            r = requests.post(
                f'{API_BASE}/translate/batch/',
                headers=headers,
                json=payload,
                timeout=120,
            )
        except requests.exceptions.RequestException as e:
            self.stdout.write(self.style.ERROR(f'  [{lang_str}] Request error: {e}'))
            if _retry < 3:
                wait = 15 * (_retry + 1)
                self.stdout.write(f'  [{lang_str}] Retrying in {wait}s (attempt {_retry + 2}/4)')
                time.sleep(wait)
                return self._translate_batch_multi(texts, target_languages, _retry=_retry + 1)
            return None

        # Synchronous response (small batches may return directly)
        if r.status_code == 200:
            data = r.json()
            translations = data.get('translations', [])
            if translations and not isinstance(translations[0], dict):
                lang = target_languages[0]
                return [{lang: t} for t in translations]
            return translations

        # Async job — poll until done, trust the API status
        if r.status_code == 202:
            resp_data = r.json()
            job_id = resp_data.get('job_id')
            if not job_id:
                return None

            # Use poll_url from response (absolute URL), fall back to constructed URL
            poll_url = resp_data.get('poll_url') or f'{API_BASE}/jobs/{job_id}/'
            if poll_url.startswith('/'):
                poll_url = f'{API_BASE}/jobs/{job_id}/'  # safety: reject relative paths
            queue_pos = resp_data.get('queue_position')
            if queue_pos:
                self.stdout.write(f'  [{lang_str}] Queued (position {queue_pos})')

            last_log = 0
            poll_start = time.time()
            max_poll_time = MAX_POLL_TIME

            while True:
                time.sleep(POLL_INTERVAL)
                try:
                    poll_r = requests.get(poll_url, headers=headers, timeout=60)
                    if poll_r.status_code != 200:
                        continue
                    data = poll_r.json()
                    status = data.get('status')
                    processed = data.get('processed_texts', 0)

                    if status == 'completed':
                        result = data.get('result_data', {})
                        translations = result.get('translations', [])
                        if translations and not isinstance(translations[0], dict):
                            lang = target_languages[0]
                            return [{lang: t} for t in translations]
                        return translations
                    elif status == 'failed':
                        err = data.get('error_message', 'unknown')
                        # Transient errors — retry the whole job
                        if 'restart' in err.lower() or 'stall' in err.lower() or 'cancel' in err.lower():
                            if _retry < 3:
                                wait = 15 * (_retry + 1)
                                self.stdout.write(self.style.WARNING(
                                    f'  [{lang_str}] Transient: {err} — retrying in {wait}s (attempt {_retry + 2}/4)'
                                ))
                                time.sleep(wait)
                                return self._translate_batch_multi(texts, target_languages, _retry=_retry + 1)
                        self.stdout.write(self.style.ERROR(f'  [{lang_str}] Job failed: {err}'))
                        return None

                    # Timeout — keep polling same job, don't resubmit (API recommendation)
                    if time.time() - poll_start > max_poll_time:
                        self.stdout.write(self.style.WARNING(
                            f'  [{lang_str}] Job timed out after {max_poll_time}s (job_id={job_id})'
                        ))
                        return None

                    # Log progress every 15s
                    now = time.time()
                    if now - last_log > 15:
                        pct = data.get('progress_percentage', 0)
                        total = data.get('total_texts', 0)
                        elapsed = int(now - poll_start)
                        pos = data.get('queue_position', '')
                        pos_str = f' q:{pos}' if pos else ''
                        self.stdout.write(f'  [{lang_str}] {status} {pct:.0f}% ({processed}/{total}){pos_str} [{elapsed}s]')
                        last_log = now

                except requests.exceptions.RequestException as e:
                    self.stdout.write(self.style.WARNING(f'  [{lang_str}] Poll error: {e}'))
                    continue

        # Handle rate limiting
        if r.status_code == 429:
            self.stdout.write(self.style.WARNING(f'  [{lang_str}] Rate limited — waiting 30s'))
            time.sleep(30)
            return self._translate_batch_multi(texts, target_languages, _retry=_retry)

        # Handle quota exceeded
        if r.status_code == 402:
            self.stdout.write(self.style.ERROR(f'  Quota exceeded! {r.text[:200]}'))
            return None

        self.stdout.write(self.style.ERROR(
            f'  [{lang_str}] HTTP {r.status_code}: {r.text[:200]}'
        ))
        return None
