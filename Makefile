.PHONY: ci install check format-check lint types-check test e2e-smoke install-windows-start-menu uninstall-windows-start-menu mobile-start mobile-android mobile-ios mobile-web mobile-lint mobile-types-check

ci:
	npm ci

start:
	npm run dev

install:
	npm install

format:
	npm run format

format-check:
	npm run format-check

lint:
	npm run lint

types-check:
	npm run types-check

test:
	npm run test

e2e-smoke:
	npm run e2e:smoke

install-windows-start-menu:
	powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install-windows-start-menu.ps1

uninstall-windows-start-menu:
	powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install-windows-start-menu.ps1 -Uninstall

mobile-start:
	npm run mobile:start

mobile-android:
	npm run mobile:android

mobile-ios:
	npm run mobile:ios

mobile-web:
	npm run mobile:web

mobile-lint:
	npm run mobile:lint

mobile-types-check:
	npm run mobile:types-check

check:
	npm run check

# BEGIN ELENTHOS SKILLS
.PHONY: skill-update skill-update-sources skill-update-elenthos skill-publish-elenthos skill-publish-elenthos-dry-run
PYTHON ?= python
SKILL_PUBLISH_MESSAGE ?= Update Elenthos skills

skill-update-sources:
	$(PYTHON) .codex/skills/elenthos/elenthos-skills/elenthos-skills-update/scripts/update_skill_bundles.py update

skill-update-elenthos:
	$(PYTHON) .codex/skills/elenthos/elenthos-skills/elenthos-skills-sync/scripts/sync_elenthos_skills.py update

skill-update: skill-update-sources skill-update-elenthos

skill-publish-elenthos:
	$(PYTHON) .codex/skills/elenthos/elenthos-skills/elenthos-skills-sync/scripts/sync_elenthos_skills.py publish --stage-parent --message "$(SKILL_PUBLISH_MESSAGE)"

skill-publish-elenthos-dry-run:
	$(PYTHON) .codex/skills/elenthos/elenthos-skills/elenthos-skills-sync/scripts/sync_elenthos_skills.py publish --dry-run --message "$(SKILL_PUBLISH_MESSAGE)"
# END ELENTHOS SKILLS
