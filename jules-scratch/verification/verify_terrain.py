import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Go to the Streamlit app
        await page.goto("http://localhost:8501")

        # Wait for the app to load
        await expect(page.get_by_text("Confronto Semantico 3D tra Testi")).to_be_visible(timeout=180000)

        # Input text into the text areas
        await page.get_by_label("Testo 1").fill("Questo è il primo testo. Parla di intelligenza artificiale.")
        await page.get_by_label("Testo 2").fill("Questo è il secondo testo. Discute di apprendimento automatico.")

        # Click the analyze button
        button = page.get_by_text("Analizza testi")
        await button.click()

        # Wait for a long time for the analysis to complete
        await page.wait_for_timeout(180000)

        # Wait for the 3D plot to appear
        await expect(page.get_by_text("Confronto tra Paesaggi Semantici")).to_be_visible()

        # Take a screenshot
        await page.screenshot(path="jules-scratch/verification/verification.png")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())