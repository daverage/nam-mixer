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
