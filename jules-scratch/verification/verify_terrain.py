from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://localhost:8501")

    # Wait for the app to load
    page.wait_for_selector("text=Quanti testi vuoi confrontare?")

    # Fill in the text areas
    page.fill("textarea[aria-label='Testo 1']", "This is the first text. It is about nature and mountains.")
    page.fill("textarea[aria-label='Testo 2']", "This is the second text. It is about rivers and lakes.")

    # Click the analyze button
    page.click("text=Analizza testi")

    # Wait for the terrain to be generated
    page.wait_for_selector("text=Paesaggi Semantici", timeout=120000)

    # Take a screenshot
    page.screenshot(path="jules-scratch/verification/verification.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)