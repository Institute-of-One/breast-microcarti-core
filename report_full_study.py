"""Report actual held-out results and fixed-opacity US/SR overlay."""
import json
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_models import StudyModel
import study_budget as budget

OUT=Path(__file__).resolve().parent/'results/full-study'

def main():
    start=time.perf_counter()
    try:
        budget.check()
        result=json.loads((OUT/'evaluation.json').read_text())
        rows=json.loads((OUT/'phantoms.json').read_text())
        norm=json.loads((OUT/'normalization.json').read_text())
        images=np.load(OUT/'images.npy',mmap_mode='r'); targets=np.load(OUT/'targets.npy',mmap_mode='r')
        # First held-out positive by ID, fixed 3MHz/full/40dB; no visually selected winner.
        i=next(r['id'] for r in rows if r['split']=='test' and r['class']==1)
        x=images[i,1,2].copy(); z=np.clip((x-norm['median'])/norm['iqr'],-5,5)
        model=StudyModel('xray_sr')
        model.load_state_dict(torch.load(OUT/'xray_sr.pt',map_location='cpu',weights_only=True)['state']); model.eval()
        torch.set_num_threads(1)
        with torch.no_grad(): _,s=model(torch.from_numpy(z[None,None])); prob=s.sigmoid().numpy()[0,0]
        fig,axes=plt.subplots(1,4,figsize=(14,3.7))
        axes[0].imshow(z,cmap='gray',vmin=-5,vmax=5); axes[0].set_title('US reconstruction')
        axes[1].imshow(prob,cmap='magma',vmin=0,vmax=1); axes[1].set_title('US-derived SR probability')
        axes[2].imshow(z,cmap='gray',vmin=-5,vmax=5,extent=(0,256,256,0)); axes[2].imshow(prob,cmap='magma',vmin=0,vmax=1,alpha=.35); axes[2].set_title('Overlay (opacity 0.35)')
        axes[3].imshow(targets[i],cmap='gray',vmin=0,vmax=1); axes[3].set_title('Synthetic truth (reference)')
        for ax in axes: ax.axis('off')
        fig.suptitle(f'Fixed held-out phantom {i}: 3 MHz, full view, 40 dB; no clinical X-ray input')
        fig.tight_layout(); fig.savefig(OUT/'heldout-overlay.png',dpi=160); plt.close(fig)
        text=['# 全例・3経路比較の実測結果','',
          '480独立phantom、3,840再構成画像。train/validation/testはphantom単位で分離。各経路30epoch、validation最良checkpoint。以下は合成100 µm・既定配置の結果であり、実患者の悪性分類やマンモグラフィ代替性を示さない。','']
        for c in result['conditions']:
            text.append(f"## {c['frequency_hz']/1e6:g} MHz / {c['view_noise']}")
            for method,m in c['models'].items():
                a,s,f,p=m['metrics']; lo,hi=m['ci95'][0]
                text.append(f'- {method}: AUC {a:.4f} (95% CI {lo:.4f}–{hi:.4f}); 感度 {s:.3f} / test FPR {f:.3f}; 配置balanced accuracy {p:.3f}。')
            for name,d in c['paired_differences'].items():
                lo,hi=d['ci95'][0]; text.append(f"- {name}: ΔAUC {d['metrics'][0]:.4f} ({lo:.4f}–{hi:.4f})。")
            loc=c['sr_froc'][4]; text.append(f"- SR局在閾値0.5: 感度 {loc['sensitivity']:.3f}, FP/画像 {loc['fp_per_image']:.3f}。")
            text.append('')
        text+=['## 成立性判断','',
          '技術的実行：全例再構成と固定3経路比較まで完了。物理・数値gateと資源台帳は別の証拠として保存。',
          '研究範囲：対応付き臨床微小石灰化参照の適格性は未成立のため、合成の機序比較へ縮小を維持。画像の見た目を成功基準にしない。',
          '科学的差別化・投稿可能性：既知背景、2D円柱、単一径・単一seed、理想化した雑音と配置に限定。AI出力だけから検出不能性や理想観測者限界は証明できず、この比較単独で元の検出限界論文が成立したとは判定しない。追加モデルで弱い効果を救済しない。',
          '独立性：testはモデル選択に未使用。今回のtestは評価消費済みであり、次段階の未使用testとして再利用しない。外部AI独立レビューは送信承認がないためNOT_RUN。','',
          f'最終report開始前の累積計算量：{budget.consumed()/3600:.4f} device-hours。report等を含む最終量はresource.jsonl / pipeline-complete.json参照。',
          '詳細の固定threshold・混同行列・paired bootstrap・noise-only対照は evaluation.json、実行時間は各training JSONとresource.jsonlに保存。']
        (OUT/'DECISION.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    finally: budget.record('cpu_final_report',time.perf_counter()-start)

if __name__=='__main__': main()
