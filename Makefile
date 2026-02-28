.PHONY: ci install check format-check lint types-check test

ci:
	npm ci

install:
	npm install

format-check:
	npm run format-check

lint:
	npm run lint

types-check:
	npm run types-check

test:
	npm run test

check:
	npm run check
