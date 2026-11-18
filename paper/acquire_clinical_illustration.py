"""Authorized public clinical illustration; bounded ZIP ranges, no external upload."""
import sys,json,zipfile,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_bcmid import RemoteZip

def main():
    folder=ROOT/'paper/sources'; folder.mkdir(exist_ok=True)
    with urllib.request.urlopen('https://zenodo.org/api/records/14859694',timeout=30) as f: record=json.load(f)
    assert record['metadata']['license']['id']=='cc-by-4.0' and record['metadata']['access_right']=='open'
    member=next(x for x in record['files'] if x['key']=='Dataset_v3.zip')
    remote=RemoteZip(member['links']['self'],member['size'])
    with zipfile.ZipFile(remote) as z:
        entries=[{'name':x.filename,'size':x.file_size,'compressed':x.compress_size} for x in z.infolist()]
        (folder/'clinical-zip-directory.local.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
        if '--fetch' not in sys.argv:
            names=[e for e in entries if 'suspicious' in e['name'].lower()][:22]
            print(json.dumps(names,indent=2)); print('transferred bytes',remote.transferred); return
        # Select the first lexically ordered recent CC DICOM in the suspicious group.
        candidates=sorted(e['name'] for e in entries if 'suspicious' in e['name'].lower() and e['name'].lower().endswith('.dcm') and 'cc' in Path(e['name']).name.lower() and 'recent' in Path(e['name']).name.lower())
        if not candidates: raise RuntimeError('No deterministic eligible clinical image')
        selected=candidates[0]; stem=Path(selected).stem
        associated=[e['name'] for e in entries if str(Path(e['name']).parent)==str(Path(selected).parent) and Path(e['name']).suffix.lower() in ('.jpg','.jpeg','.png') and 'cc' in Path(e['name']).name.lower() and 'recent' in Path(e['name']).name.lower()]
        cache=ROOT/'.cache/public-clinical'; cache.mkdir(exist_ok=True,parents=True)
        provenance={'source_record':'https://zenodo.org/records/14859694','license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/','use':'Clinical-context illustration only; not AI training, validation or testing','selection':'Lexically first suspicious-group recent CC DICOM; not chosen based on model performance','files':[]}
        for name,local in [(selected,'clinical.dcm')]+([(sorted(associated)[0],'annotation.jpg')] if associated else []):
            info=z.getinfo(name)
            if info.file_size>100*1024**2: raise RuntimeError('Uncompressed member exceeds illustration cap')
            data=z.read(name); (cache/local).write_bytes(data)
            provenance['files'].append({'archive_member':name,'local_name':local,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        provenance['transferred_zip_bytes']=remote.transferred
        provenance['creators']=record['metadata'].get('creators'); provenance['title']=record['metadata']['title']; provenance['doi']=record['metadata'].get('doi')
        (folder/'clinical-illustration-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in provenance.items() if k not in ('creators',)},indent=2))

if __name__=='__main__': main()
