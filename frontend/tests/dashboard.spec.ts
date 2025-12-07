# F1 Strategy Dashboard - Sample E2E Test

import { test, expect } from '@playwright/test';

// Helper function to login
async function login(page: any, username: string, password: string) {
  await page.goto('/login');
  await page.fill('input[id="username"]', username);
  await page.fill('input[id="password"]', password);
  await page.click('button[type="submit"]');
  await page.waitForURL('/strategy');
}

test.describe('Authentication', () => {
  test('should login with valid credentials', async ({ page }) => {
    await page.goto('/login');
    
    // Fill login form
    await page.fill('input[id="username"]', 'admin');
    await page.fill('input[id="password"]', 'admin123');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Should redirect to strategy page
    await expect(page).toHaveURL('/strategy');
  });

  test('should show error with invalid credentials', async ({ page }) => {
    await page.goto('/login');
    
    // Fill with invalid credentials
    await page.fill('input[id="username"]', 'invalid');
    await page.fill('input[id="password"]', 'wrong');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Should show error message
    await expect(page.locator('text=Login failed')).toBeVisible();
  });
});

test.describe('Live Strategy Console', () => {
  test.beforeEach(async ({ page }) => {
    await login(page, 'admin', 'admin123');
  });

  test('should display page title', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Live Strategy Console');
  });

  test('should have driver selector', async ({ page }) => {
    await expect(page.locator('select')).toBeVisible();
  });
});

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await login(page, 'admin', 'admin123');
  });

  test('should navigate to all pages', async ({ page }) => {
    // Navigate to Lap Monitor
    await page.click('text=Lap Monitor');
    await expect(page).toHaveURL('/lap-monitor');
    await expect(page.locator('h1')).toContainText('Lap-by-Lap Monitor');

    // Navigate to Strategy Tree
    await page.click('text=Strategy Tree');
    await expect(page).toHaveURL('/strategy-tree');
    await expect(page.locator('h1')).toContainText('Strategy Tree Visualizer');

    // Navigate to Competitors
    await page.click('text=Competitors');
    await expect(page).toHaveURL('/competitors');
    await expect(page.locator('h1')).toContainText('Competitor Comparison');

    // Navigate to Weather
    await page.click('text=Weather');
    await expect(page).toHaveURL('/weather');
    await expect(page.locator('h1')).toContainText('Weather & Track Evolution');
  });
});
