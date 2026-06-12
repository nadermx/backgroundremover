import os
import re

from django.conf import settings
from django.core.management import BaseCommand

from translations.models.textbase import TextBase
from translations.models.translation import Translation


class Command(BaseCommand):
    help = 'Scan templates for i18n keys and seed TextBase entries'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without making changes',
        )
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Mark all existing TextBase entries as untranslated (forces re-translation)',
        )

    def handle(self, *args, **options):
        dry_run = options.get('dry_run', False)
        reset = options.get('reset', False)

        if reset:
            count = TextBase.objects.filter(translated=True).update(translated=False)
            self.stdout.write(self.style.WARNING(f'Reset {count} entries to untranslated'))

        all_keys = self._scan_templates()
        self.stdout.write(f'\nTotal unique keys: {len(all_keys)}\n')

        if dry_run:
            prefixes = {}
            for code_name in sorted(all_keys.keys()):
                parts = code_name.split('_')
                group = '_'.join(parts[:2]) if len(parts) > 2 else parts[0]
                prefixes.setdefault(group, []).append(code_name)

            for group in sorted(prefixes.keys()):
                keys = prefixes[group]
                self.stdout.write(f'\n  [{group}] ({len(keys)} keys)')
                for code_name in keys[:3]:
                    text = all_keys[code_name]
                    preview = text[:60] + '...' if len(text) > 60 else text
                    self.stdout.write(f'    {code_name}: {preview}')
                if len(keys) > 3:
                    self.stdout.write(f'    ... and {len(keys) - 3} more')
            return

        created = 0
        updated = 0
        skipped = 0

        for code_name, text in sorted(all_keys.items()):
            try:
                obj = TextBase.objects.get(code_name=code_name)
                if obj.text != text:
                    obj.text = text
                    obj.translated = False
                    obj.save()
                    updated += 1
                    self.stdout.write(f'  Updated: {code_name}')
                else:
                    skipped += 1
            except TextBase.DoesNotExist:
                TextBase.objects.create(
                    code_name=code_name,
                    text=text,
                    translated=False
                )
                created += 1
                self.stdout.write(f'  Created: {code_name}')

        # Sync English translations
        en_created = 0
        for code_name, text in all_keys.items():
            try:
                en_obj = Translation.objects.get(code_name=code_name, language='en')
                if en_obj.text != text:
                    en_obj.text = text
                    en_obj.save()
            except Translation.DoesNotExist:
                Translation.objects.create(
                    code_name=code_name,
                    language='en',
                    text=text
                )
                en_created += 1

        self.stdout.write(self.style.SUCCESS(
            f'\nDone! Created: {created}, Updated: {updated}, Skipped: {skipped}'
        ))
        self.stdout.write(f'English translations created: {en_created}')
        self.stdout.write(f'Total TextBase entries: {TextBase.objects.count()}')
        untranslated = TextBase.objects.filter(translated=False).count()
        if untranslated:
            self.stdout.write(self.style.WARNING(
                f'{untranslated} entries pending translation. Run: python manage.py run_translation'
            ))

    def _scan_templates(self):
        """Scan HTML templates for g.i18n.CODE_NAME|default:'TEXT' patterns."""
        base_dir = settings.BASE_DIR
        template_dirs = [os.path.join(base_dir, 'templates')]

        pattern = re.compile(
            r'(?:g\.)?i18n\.(\w+)\|default:(["\'])(.+?)\2',
            re.DOTALL
        )

        keys = {}
        files_scanned = 0

        for template_dir in template_dirs:
            if not os.path.exists(template_dir):
                continue
            for root, dirs, files in os.walk(template_dir):
                for filename in files:
                    if not filename.endswith('.html'):
                        continue
                    filepath = os.path.join(root, filename)
                    files_scanned += 1
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    matches = pattern.findall(content)
                    for code_name, _quote, text in matches:
                        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                        if code_name not in keys:
                            keys[code_name] = text

        self.stdout.write(f'Scanned {files_scanned} template files, found {len(keys)} keys')
        return keys
