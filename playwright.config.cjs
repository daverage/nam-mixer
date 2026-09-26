const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  testMatch: 'accessibility.spec.cjs',
  use: { baseURL: 'http://127.0.0.1:5001', browserName: 'chromium' },
  webServer: {
    command: 'python3 app.py',
    url: 'http://127.0.0.1:5001/',
    timeout: 90000,
    reuseExistingServer: !process.env.CI,
  },
});
