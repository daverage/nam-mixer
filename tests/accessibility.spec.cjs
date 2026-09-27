const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

test('keyboard entry, blocked workflow, and Continuous Gain navigation', async ({ page }) => {
  await page.goto('/');
  const welcome = page.getByRole('dialog', { name: 'Welcome to NAM Mixer' });
  await expect(welcome).toBeVisible();
  await expect(welcome).toContainText('Choose two amps');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: 'Get started' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(welcome).toBeHidden();
  await expect(page.getByRole('button', { name: 'Builder', exact: true })).toBeFocused();

  const compare = page.getByRole('button', { name: /Compare amps/ });
  await compare.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#workflow-hint')).toContainText('Prepare your amps first');
  await expect(compare).toBeFocused();

  const cg = page.getByRole('button', { name: 'Continuous Gain', exact: true });
  await cg.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#cg-title')).toBeFocused();
  await expect(cg).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'New project' }).click();
  await expect(page.getByRole('textbox', { name: 'New project name' })).toBeFocused();
});

test('key controls expose names and current values', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('nam-mixer-welcome-seen', '1'));
  await page.goto('/');
  await expect(page.getByLabel('Amp A NAM file')).toHaveAttribute('type', 'file');
  await expect(page.getByLabel('Amp B NAM file')).toHaveAttribute('type', 'file');
  const slider = page.locator('#crossover-knob-slider');
  await expect(slider).toHaveAttribute('aria-valuetext', /out of 10/);
});

test('core controls have no detectable name or ARIA violations', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('nam-mixer-welcome-seen', '1'));
  await page.goto('/');
  const builder = await new AxeBuilder({ page }).withRules([
    'label', 'button-name', 'aria-allowed-attr', 'aria-required-attr', 'aria-valid-attr-value', 'duplicate-id',
  ]).analyze();
  expect(builder.violations).toEqual([]);
  await page.getByRole('button', { name: 'Continuous Gain', exact: true }).click();
  const cg = await new AxeBuilder({ page }).include('#cg-panel').withRules([
    'label', 'button-name', 'aria-allowed-attr', 'aria-required-attr', 'aria-valid-attr-value', 'duplicate-id',
  ]).analyze();
  expect(cg.violations).toEqual([]);
});

test('large result panels are quiet and summaries go through one announcer', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('nam-mixer-welcome-seen', '1'));
  await page.goto('/');
  const announcer = page.locator('#sr-announcer');
  await expect(announcer).toHaveAttribute('role', 'status');
  await expect(announcer).toHaveAttribute('aria-atomic', 'true');
  for (const id of ['recipe-result', 'tone3000-results', 'generate-result', 'kaggle-result', 'local-result', 'tool-inspector-result', 'session-list']) {
    await expect(page.locator(`#${id}`)).not.toHaveAttribute('aria-live', /.+/);
  }
  await page.evaluate(() => announce('Training finished. Technical validation passed.'));
  await expect(announcer).toHaveText('Training finished. Technical validation passed.');
});

test('journey chart has a text summary and data table', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('nam-mixer-welcome-seen', '1'));
  await page.goto('/');
  await page.evaluate(() => renderJourneyText({
    times: [0, 1, 2, 3, 4],
    envelope_db: [-60, -30, -20, -10, -5],
    blend_weight: [0, 0.05, 0.5, 0.95, 1],
  }));
  const summary = page.locator('#journey-summary');
  await expect(summary).toContainText('mostly Amp A 25% of the time, changing between amps 25%, and mostly Amp B 50%');
  await expect(summary).toContainText('The changeover starts at 2.0 s and first reaches mostly Amp B at 3.0 s.');
  const rows = page.locator('#journey-data-rows tr');
  await expect(rows).toHaveCount(5);
  await expect(rows.first()).toContainText('Silent');
  await expect(rows.nth(3)).toContainText('Mostly Amp B');
});
