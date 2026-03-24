.PHONY: ci install check format-check lint types-check test

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

check:
	npm run check
