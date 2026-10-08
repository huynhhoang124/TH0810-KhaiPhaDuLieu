# Xuất notebook đã chạy -> PDF. Kết quả mỗi ô code được CHỤP MÀN HÌNH (chỉ phần output, không có code)
# từ notebook render trong Chrome, lưu vào screenshots/, rồi chèn vào báo cáo dưới dạng ảnh có chú thích.
import subprocess, sys, pathlib, shutil
from playwright.sync_api import sync_playwright

nb = pathlib.Path(sys.argv[1]).resolve(); root = nb.parent
full, rep, pdf = root / '_full.html', root / '_report.html', root / 'BaoCao_TH0810.pdf'
shots = root / 'screenshots'; shutil.rmtree(shots, ignore_errors=True); shots.mkdir()

def convert(out, *flags):
    subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'html', *flags, str(nb),
                    '--output', out.name], check=True, capture_output=True)

convert(full)                                   # giao diện Jupyter đầy đủ (để chụp output)
convert(rep, '--no-input', '--no-prompt')       # khung báo cáo (markdown)

SHOT_CSS = """.jp-OutputArea-output, .jp-RenderedHTMLCommon { overflow: visible !important; }
body { background: white; } .jp-OutputArea-child { width: fit-content !important; max-width: 100%; padding: 6px 10px 6px 0; }"""
REPORT_CSS = """<style>
@page { size: A4; margin: 14mm 12mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt; line-height: 1.45; }
.jp-Notebook, body { padding: 0 !important; max-width: none !important; }
h1 { margin-top: 22px; border-bottom: 2px solid #4C72B0; padding-bottom: 4px; }
h2 { color: #2d4f8a; margin-top: 18px; } h3 { color: #3a3a3a; }
h1, h2, h3, .jp-RenderedMarkdown p:only-child { break-after: avoid; page-break-after: avoid; }
table { font-size: 8.5pt !important; border-collapse: collapse; margin: 6px 0 !important; }
th, td { padding: 2px 6px !important; border: 1px solid #ccc !important; }
figure.shot { margin: 10px 0 14px; page-break-inside: avoid; text-align: center; }
figure.shot img { max-width: 100%; border: 1px solid #b9c2cf; border-radius: 4px;
                  box-shadow: 0 1px 4px rgba(0,0,0,.18); }
figure.shot figcaption { font-size: 9pt; font-style: italic; color: #555; margin-top: 4px; }
</style>"""

with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome')
    # 1) Chụp màn hình output của từng ô code
    pg = b.new_page(viewport={'width': 1000, 'height': 900}, device_scale_factor=2)
    pg.goto(full.as_uri(), wait_until='load'); pg.add_style_tag(content=SHOT_CSS); pg.wait_for_timeout(1000)
    files, n = [], 0   # mỗi ô code -> danh sách ảnh cho từng output con (None = giữ nguyên markdown)
    for cell in pg.locator('.jp-CodeCell').all():
        cur = []
        for child in cell.locator('.jp-OutputArea-child').all():
            box = child.bounding_box()
            if child.locator('.jp-RenderedMarkdown').count() or not box or box['height'] < 12:
                cur.append(None); continue
            n += 1; f = shots / f'ket_qua_{n:02d}.png'
            child.screenshot(path=str(f)); cur.append(f.as_uri())
        files.append(cur)
    print('Đã chụp', n, 'ảnh màn hình kết quả')

    # 2) Thay output trong báo cáo bằng ảnh chụp + chú thích "Hình n"
    pg = b.new_page()
    pg.goto(rep.as_uri(), wait_until='load')
    n_cells = pg.evaluate("""(files) => {
      const clean = t => t.replace('¶', '').trim();
      let head = '', n = 0, code = 0;
      for (const cell of document.querySelectorAll('.jp-Cell')) {
        if (cell.classList.contains('jp-MarkdownCell')) {
          const hs = cell.querySelectorAll('h1, h2, h3'); if (hs.length) head = clean(hs[hs.length - 1].textContent);
          continue;
        }
        const fs = files[code++] || [];
        const kids = [...cell.querySelectorAll('.jp-OutputArea-child')];
        kids.forEach((k, j) => {
          const f = fs[j];
          if (!f) { if (!k.querySelector('.jp-RenderedMarkdown')) k.remove(); return; }
          n++;
          k.outerHTML = `<figure class="shot"><img src="${f}"><figcaption>Hình ${n}. ${head} – ảnh chụp màn hình kết quả chạy chương trình</figcaption></figure>`;
        });
      }
      // trang bìa tách riêng
      const first = document.querySelector('.jp-Cell'); first.style.pageBreakAfter = 'always';
      document.querySelectorAll('h1')[0].style.pageBreakBefore = 'auto';
      return code;
    }""", files)
    assert n_cells == len(files), (n_cells, len(files))
    pg.add_style_tag(content=REPORT_CSS.replace('<style>', '').replace('</style>', ''))
    pg.evaluate("Promise.all([...document.images].map(i => i.complete ? 0 : new Promise(r => i.onload = i.onerror = r)))")
    # ảnh chụp ở tỉ lệ 2x -> hiển thị bằng nửa kích thước pixel
    pg.evaluate("document.querySelectorAll('figure.shot img').forEach(i => i.style.width = (i.naturalWidth / 2) + 'px')")
    pg.pdf(path=str(pdf), format='A4', print_background=True,
           margin={'top': '14mm', 'bottom': '14mm', 'left': '12mm', 'right': '12mm'},
           display_header_footer=True, header_template='<span></span>',
           footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#888">Trang <span class=pageNumber></span> / <span class=totalPages></span></div>')
    b.close()

full.unlink(); rep.unlink()
print('PDF:', pdf, pdf.stat().st_size // 1024, 'KB')
