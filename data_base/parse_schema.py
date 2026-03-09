import zipfile, re

with zipfile.ZipFile('brain_tumor.mwb', 'r') as z:
    with z.open('document.mwb.xml') as f:
        content = f.read().decode('utf-8')

chunks = re.split(r'struct-name="db\.mysql\.Table"', content)
for chunk in chunks[1:]:
    tname = re.search(r'<value type="string" key="name">([^<]+)</value>', chunk)
    if not tname:
        continue
    print('Table:', tname.group(1))
    cols = re.findall(r'struct-name="db\.mysql\.Column".*?<value type="string" key="name">([^<]+)</value>', chunk[:8000], re.DOTALL)
    dtypes = re.findall(r'mysql\.datatype\.([a-z]+)', chunk[:8000])
    for j, col in enumerate(cols):
        dt = dtypes[j] if j < len(dtypes) else '?'
        print('  ', col, ':', dt)
    print()
