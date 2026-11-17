"""Describe the completed targeted diagnostic using CSV/JSON only; no models."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,statistics

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
OUT=BASE/'training/runs/p1_v2_mse_mae_20261113_full/shap_convergence_20261113'
TARGET=OUT/'summary'

def require(v,msg):
    if not v:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def num(v):return f'{v:.6f}'.replace('.',',')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute-summary',action='store_true');args=ap.parse_args()
    receipt=read(OUT/'convergence_receipt.json');plan=read(HERE/'locked_convergence_plan.json')
    require(receipt['status']=='SUCCESS_FIXED_MODEL_TARGETED_SHAP_CONVERGENCE','Execution incomplete')
    require(receipt['plan_sha256']==sha(HERE/'locked_convergence_plan.json'),'Plan mismatch')
    require(receipt['selected_cases']==8 and receipt['metric_rows']==576 and receipt['draw_counts']==[256,1024,4096],'Scope changed')
    for name,entry in read(OUT/'OUTPUT_MANIFEST.json')['files'].items():
        p=OUT/name;require(p.stat().st_size==entry['size_bytes'] and sha(p)==entry['sha256'],'Output changed: '+name)
    if not args.execute_summary:
        print(json.dumps({'status':'READY_CSV_ONLY_NO_WRITES','cases':8}));return
    require(not TARGET.exists(),'Refuse overwrite');TARGET.mkdir()
    with (OUT/'convergence_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==576,'Incomplete metrics')
    selected=[]
    for c in plan['conditions']:
        byrepeat=[]
        for rep in ['original_path_continuation','fresh_1','fresh_2']:
            chosen=sorted([r for r in rows if r['condition']==c['condition'] and int(r['position'])==c['selected_worst_explanation_position'] and int(r['output_index'])==c['selected_worst_output_index'] and r['repeat']==rep],key=lambda x:int(x['nsamples']))
            require([int(r['nsamples']) for r in chosen]==[256,1024,4096],'Missing selected diagnostic rows')
            byrepeat.append({'repeat':rep,'target_date':chosen[0]['target_date'],
                'residual_scaled_by_draw_count':{r['nsamples']:float(r['residual_scaled']) for r in chosen},
                'background_residual_scaled_by_draw_count':{r['nsamples']:float(r['background_sampling_residual_scaled']) for r in chosen},
                'path_residual_scaled_by_draw_count':{r['nsamples']:float(r['path_sampling_residual_scaled']) for r in chosen},
                'raw_residual_by_draw_count':{r['nsamples']:float(r['residual_raw']) for r in chosen},
                'absolute4096_less_than_absolute256':abs(float(chosen[2]['residual_scaled']))<abs(float(chosen[0]['residual_scaled'])),
                'absolute_residual_monotone_nondecreasing':abs(float(chosen[0]['residual_scaled']))<=abs(float(chosen[1]['residual_scaled']))<=abs(float(chosen[2]['residual_scaled'])),
                'absolute_residual_monotone_nonincreasing':abs(float(chosen[0]['residual_scaled']))>=abs(float(chosen[1]['residual_scaled']))>=abs(float(chosen[2]['residual_scaled']))})
        selected.append({'arm':c['arm'],'condition':c['condition'],'output':c['selected_worst_output'],
            'explanation_position':c['selected_worst_explanation_position'],'repeats':byrepeat})
    pooled=[]
    for n in (256,1024,4096):
        rr=[r for r in rows if int(r['nsamples'])==n];v=[float(r['abs_residual_scaled']) for r in rr]
        pooled.append({'nsamples':n,'dependent_case_output_repeat_rows':len(v),'mean_abs_scaled_residual':statistics.fmean(v),
            'median_abs_scaled_residual':statistics.median(v),'max_abs_scaled_residual':max(v),
            'scope':'Only outcome-selected conditions and four chronological dates each; descriptive, not whole-test accuracy'})
    lines=['13 Kasım 2026 — Seçilmiş örneklerde SHAP çekiliş duyarlılığı',
        'Bu bir model eğitimi veya yeni model seçimi değildir. Mevcut MSE ve MAE modellerinin birer koşulunda, en büyük önceki yerel artık ile test örnekleminin başı/ortası/sonu incelendi. Toplam8 pencere-koşul, bütün8çıktı, üç çekiliş akışı ve256/1024/4096çekiliş vardır.',
        'Örnek seçimi büyük artıklar görüldükten sonra yapıldı. Bu nedenle sonuçlar hedefli sayısal tanıdır; 128 veya584 tarihin geneline ait temsilci bir hata tahmini değildir.',
        'Her akışta aynı4096yolun256/1024/4096ön ekleri kullanıldı. Özgün akışın ilk256yolu önceki açıklamayla birebir aynıdır; ayrıca iki yeni tohum vardır. Akışlar ve tarihlerden p-değeri veya bağımsız tekrar iddiası üretilmedi. Artığın her adımda azalacağı varsayılmadı.',
        '', 'Seçilmiş uç örnekler — işaretli artıklar, eğitim hedef standart sapması biriminde:',
        'Koşul / çıktı / tarih | Akış | 256 | 1024 | 4096']
    for c in selected:
        for r in c['repeats']:
            vals=r['residual_scaled_by_draw_count']
            lines.append(f"{c['arm'].upper()} / {c['output']} / {r['target_date']} | {r['repeat']} | {num(vals['256'])} | {num(vals['1024'])} | {num(vals['4096'])}")
    lines += ['', 'Artığın ayrıştırılması:',
        'R_toplam=f(x)−ortalama64 f(arka plan)−katkılar toplamı.',
        'R_arka_plan=çekilen arka plan tahminlerinin ortalaması−ortalama64 f(arka plan).',
        'R_yol=f(x)−çekilen arka plan tahminlerinin ortalaması−katkılar toplamı.',
        'Dolayısıyla R_toplam=R_arka_plan+R_yol. Bileşenler işaretlidir; birbirini kısmen dengeleyebilir. Çekiliş sayısına bağlı bu hesaplama duyarlılığı modelin tahmin hatası değildir.',
        'Özgün akışta uç örneklerin4096çekiliş ayrışması:']
    for c in selected:
        r=c['repeats'][0]
        lines.append(f"{c['arm'].upper()} / {c['output']}: toplam {num(r['residual_scaled_by_draw_count']['4096'])} = arka plan {num(r['background_residual_scaled_by_draw_count']['4096'])} + yol {num(r['path_residual_scaled_by_draw_count']['4096'])}.")
    lines += ['', 'Sekiz seçilmiş koşul-penceredeki bütün çıktı ve akışların betimsel özeti:',
        'Çekiliş | Bağımlı satır sayısı | Ortalama mutlak artık | Medyan mutlak artık | En büyük mutlak artık']
    for r in pooled:
        lines.append(f"{r['nsamples']} | {r['dependent_case_output_repeat_rows']} | {num(r['mean_abs_scaled_residual'])} | {num(r['median_abs_scaled_residual'])} | {num(r['max_abs_scaled_residual'])}")
    lines += ['', 'Yorum ve sınır:',
        'Yukarıdaki uç örneklerin özgün ve yeni akışları birlikte değerlendirilmelidir. Daha çok çekilişle gözlenen küçülme, sonlu Monte Carlo hesabının katkısını gösterir; artığın her örnekte sıfırlandığı, her adımda monoton azaldığı veya bütün açıklamaların artık hassas olduğu sonucunu vermez.',
        'Ana128pencerelik kararlılık hesabı256çekilişte kalır. Bu tamamlayıcı kontrol onun yerine geçmez; bütün64koşul4096çekilişle yeniden çalıştırılmadı. Küçük artık elde etmek için koşul/akış seçilmedi, artık sıfıra zorlanmadı ve hiçbir eski sonuç değiştirilmedi.',
        'Aynı geçici encoder sayısal adaptörü kullanılmıştır. Kayıtlı tahminlerle sonlu uyum sınırı1e−5olarak korunmuştur; bu bütün girdilerde tam fonksiyon/türev eşitliğini kanıtlamaz. Kontrol, tarihsel veri erişilebilirliğini, tahmin üstünlüğünü veya nedenselliği doğrulamaz.',
        'Sabit model ağırlıkları ve bütün önceki korunan dosyalar değişmedi; parametre gradyanı ve optimizer adımı0. Ham katkılar, açık rastgele yollar ve uç nokta tahminleri ayrıNPZlerde;576satırlık ayrıntı convergence_metrics.csv dosyasındadır.']
    txt=TARGET/'SHAP_Cekilis_Duyarliligi_Notu_20261113.txt';txt.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary={'status':'SUCCESS_TARGETED_SHAP_CONVERGENCE_DESCRIPTIVE_SUMMARY',
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'convergence_receipt_sha256':sha(OUT/'convergence_receipt.json'),'plan_sha256':sha(HERE/'locked_convergence_plan.json'),
        'selected_worst_cases':selected,'pooled_selected_cases_only':pooled,'new_training':False,'thesis_edit':False,
        'human_note':ref(txt),'sources':[ref(p) for p in (OUT/'convergence_receipt.json',OUT/'convergence_metrics.csv',OUT/'OUTPUT_MANIFEST.json',HERE/'locked_convergence_plan.json',Path(__file__))]}
    js=TARGET/'convergence_summary.json';js.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (TARGET/'OUTPUT_MANIFEST.json').write_text(json.dumps({'status':summary['status'],'files':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in (txt,js)}},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'selected_worst_cases':selected,'pooled':pooled}))

if __name__=='__main__':main()
