.PHONY: ci install check format-check lint types-check test mobile-start mobile-android mobile-ios mobile-web mobile-lint mobile-types-check

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
