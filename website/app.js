'use strict';
let language = 'en';
let activeVoice = 0;
let activeStage = 0;
let activeMetric = 'hq';
let alignment = 'aligned';
const byId = id => document.getElementById(id);
const tr = (en, zh) => language === 'en' ? en : zh;
const voices = [
  {role:['The curious traveler','好奇的旅行者'],quote:['Iceland is so beautiful, and the vlog is very rich in content.','冰岛太美了，而且这支 vlog 的内容非常丰富。'],grounding:['A response grounded in the destinations, scenery, and travel details.','反应基于目的地介绍、风景画面与旅行细节。']},
  {role:['The feeling-first viewer','感受型观众'],quote:['I didn’t know Iceland was like this—it feels both apocalyptic and gently beautiful.','原来冰岛是这样的——既有末日般的苍凉，又有温柔的美。'],grounding:['A response to the emotional contrast between the landscapes and narration.','反应源于景观与旁白交织出的情感反差。']},
  {role:['The experienced visitor','有经验的旅人'],quote:['I’ve been to Iceland too—it’s not the easiest place to travel, but the video gives good warnings and is very well done.','我也去过冰岛——旅行并不轻松，但视频给出了很好的提醒，做得很用心。'],grounding:['A perspective that connects practical travel advice with prior experience.','从旅行建议出发，与观众的既有经验产生连接。']}
];
const stages = [
 {label:['CURATED SOCIAL SIGNALS','筛选真实社区信号'],title:['Start with what people value.','从观众在意的内容开始。'],copy:['Select 15–20 content-relevant reactions from each video’s 50 most-liked comments. A teacher model transforms them into multimodal, persona-based reasoning traces.','从每个视频的前 50 条高赞评论中，筛选 15–20 条与内容相关的反应，再由教师模型构建基于多模态内容与观众视角的推理轨迹。'],chips:[['54K pseudo-labeled traces','54K 伪标注样本'],['3K expert-labeled samples','3K 专家标注样本']]},
 {label:['SUPERVISED FINE-TUNING','监督微调'],title:['Learn to think like an audience.','学会站在观众的角度思考。'],copy:['Supervised fine-tuning teaches Qwen3-VL-8B-Think to connect content cues, viewer personas, simulated comments, and an overall quality judgment.','通过监督微调，让 Qwen3-VL-8B-Think 学习从内容线索到观众角色、模拟评论，再到总体质量判断的完整过程。'],chips:[['8B backbone','8B 基座'],['Social-CoT reasoning','Social-CoT 推理']]},
 {label:['PROCESS-SUPERVISED RL','过程监督强化学习'],title:['Make the reasoning ring true.','让推理贴近真实社区。'],copy:['GRPO uses expert-labeled samples with four rewards. Social alignment matches generated reactions to authentic comments, while diversity encourages distinct perspectives.','GRPO 使用专家标注样本与四类奖励。社会对齐奖励将生成反应与真实评论匹配，多样性奖励则鼓励不同的观众视角。'],chips:[['Format','格式'],['Label','标签'],['Diversity','多样性'],['Social alignment','社会对齐']]}
];
const models = [
 {name:'MEDEA',desc:['Our model · 8B','本文模型 · 8B'],hq:.650,macro:.749,ours:true},
 {name:'GPT-5.2',desc:['Reasoning','推理模式'],hq:.555,macro:.595},
 {name:'Claude-4.5-opus',desc:['Reasoning','推理模式'],hq:.528,macro:.522},
 {name:'Qwen3-VL-Plus',desc:['Social-CoT','Social-CoT'],hq:.508,macro:.578},
 {name:'MaxVQA',desc:['Video quality assessment','视频质量评估'],hq:.414,macro:.552},
 {name:'DOVER',desc:['Video quality assessment','视频质量评估'],hq:.339,macro:.523}
];
function renderVoice(){
 const v=voices[activeVoice];
 byId('voice-role').textContent=tr(...v.role);
 byId('voice-quote').textContent='“'+tr(...v.quote)+'”';
 byId('voice-grounding').textContent=tr(...v.grounding);
 document.querySelectorAll('[data-voice]').forEach((button,i)=>{button.setAttribute('aria-pressed',String(i===activeVoice));button.classList.toggle('active',i===activeVoice);});
}
function renderStage(){
 const stage=stages[activeStage];
 byId('stage-label').textContent=tr(...stage.label);
 byId('stage-title').textContent=tr(...stage.title);
 byId('stage-copy').textContent=tr(...stage.copy);
 byId('stage-chips').replaceChildren(...stage.chips.map(chip=>{const el=document.createElement('span');el.textContent=tr(...chip);return el;}));
 document.querySelector('.stage-numeral').textContent='0'+(activeStage+1);
 byId('training-panel').setAttribute('aria-labelledby','stage-'+activeStage);
 document.querySelectorAll('[data-stage]').forEach((button,i)=>{button.setAttribute('aria-selected',String(i===activeStage));button.tabIndex=i===activeStage?0:-1;});
}
function renderChart(){
 document.querySelectorAll('[data-metric]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.metric===activeMetric)));
 const hq=activeMetric==='hq';
 byId('headline-metric').textContent=hq?'0.650':'0.749';
 byId('headline-label').textContent=hq?tr('High-quality F1 on CASTER-Bench','CASTER-Bench 高质量类别 F1'):tr('Macro F1 on CASTER-Bench','CASTER-Bench 宏平均 F1');
 byId('headline-gain').textContent=hq?'+17.1%':'+15.4';
 byId('headline-comparison').textContent=hq?tr('relative HQ F1 improvement over GPT-5.2 (reasoning), or +9.5 points.','相较 GPT-5.2（推理模式）的 HQ F1 相对提升，绝对提升 9.5 个百分点。'):tr('points in Macro F1 over GPT-5.2 (reasoning), from 0.595 to 0.749.','相较 GPT-5.2（推理模式）的 Macro F1 绝对提升百分点，从 0.595 到 0.749。');
 const chart=byId('results-chart');
 if(!chart.children.length){models.forEach(m=>{const row=document.createElement('div');row.className='chart-row'+(m.ours?' ours':'');row.innerHTML='<div class="chart-label"></div><div class="chart-track"><div class="chart-bar"></div></div><span class="chart-value"></span>';chart.append(row);});}
 models.forEach((m,i)=>{const row=chart.children[i];row.querySelector('.chart-label').replaceChildren(document.createTextNode(m.name),Object.assign(document.createElement('small'),{textContent:tr(...m.desc)}));row.querySelector('.chart-bar').style.setProperty('--bar-width',(m[activeMetric]/.8*100)+'%');row.querySelector('.chart-value').textContent=m[activeMetric].toFixed(3);});
}
function renderCase(){
 document.querySelectorAll('[data-alignment]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.alignment===alignment)));
 const aligned=alignment==='aligned';
 const quotes=aligned?[
  [['EMOTION','情感共鸣'],['It feels both apocalyptic and gently beautiful.','既有末日般的苍凉，又有温柔的美。']],
  [['UNDERSTANDING','内容理解'],['You really understand Iceland instead of just filming the scenery.','你真的理解冰岛，而不只是拍下风景。']],
  [['PERSONAL EXPERIENCE','个人经验'],['I went to Iceland last year, and the video describes it perfectly.','我去年去过冰岛，这个视频描述得太贴切了。']]
 ]:[
  [['SIMULATED RESPONSE 01','模拟反应 01'],['So beautiful!!! It makes me really want to go!','太美了！！！真的好想去！']],
  [['SIMULATED RESPONSE 02','模拟反应 02'],['So beautiful! I really want to go.','太美了！真想去。']],
  [['SIMULATED RESPONSE 03','模拟反应 03'],['So beautiful—really want to go.','太美了——真想去。']]
 ];
 byId('case-quotes').replaceChildren(...quotes.map(([role,quote])=>{const box=document.createElement('div');box.className='case-quote';const label=document.createElement('p');label.className='quote-role';label.textContent=tr(...role);const text=document.createElement('blockquote');text.textContent='“'+tr(...quote)+'”';box.append(label,text);return box;}));
 byId('case-insight').textContent=aligned?tr('Different people notice different things. Social alignment helps preserve that specificity.','不同的人，会注意到不同的细节。社会对齐让这些具体而多元的视角得以保留。'):tr('Without alignment, the paper’s example falls back on repetitive praise. Excerpts are shortened to highlight the pattern.','缺少社会对齐时，论文中的示例退化为重复的赞美。此处精简了原始表达，以突出这一模式。');
}
function setLanguage(next){
 language=next;
 document.documentElement.lang=next==='en'?'en':'zh-CN';
 document.querySelectorAll('[data-en][data-zh]').forEach(el=>{el.innerHTML=el.dataset[next];});
 byId('language').textContent=next==='en'?'中文':'EN';
 byId('language').setAttribute('aria-label',next==='en'?'Switch to Chinese':'切换为英文');
 byId('copy-status').textContent='';
 renderVoice();renderStage();renderChart();renderCase();
 try{localStorage.setItem('caster-language',next);}catch{}
}
document.querySelectorAll('[data-voice]').forEach(button=>button.addEventListener('click',()=>{activeVoice=Number(button.dataset.voice);renderVoice();}));
document.querySelectorAll('[data-stage]').forEach(button=>{
 button.addEventListener('click',()=>{activeStage=Number(button.dataset.stage);renderStage();});
 button.addEventListener('keydown',e=>{let next=activeStage;if(e.key==='ArrowRight'||e.key==='ArrowDown')next=(next+1)%3;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')next=(next+2)%3;else if(e.key==='Home')next=0;else if(e.key==='End')next=2;else return;e.preventDefault();activeStage=next;renderStage();byId('stage-'+next).focus();});
});
document.querySelectorAll('[data-metric]').forEach(button=>button.addEventListener('click',()=>{activeMetric=button.dataset.metric;renderChart();}));
document.querySelectorAll('[data-alignment]').forEach(button=>button.addEventListener('click',()=>{alignment=button.dataset.alignment;renderCase();}));
byId('language').addEventListener('click',()=>setLanguage(language==='en'?'zh':'en'));
const menuButton=byId('menu-toggle');
function closeMenu(){menuButton.setAttribute('aria-expanded','false');byId('mobile-nav').hidden=true;}
menuButton.addEventListener('click',()=>{const open=menuButton.getAttribute('aria-expanded')!=='true';menuButton.setAttribute('aria-expanded',String(open));byId('mobile-nav').hidden=!open;});
byId('mobile-nav').querySelectorAll('a').forEach(a=>a.addEventListener('click',closeMenu));
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&menuButton.getAttribute('aria-expanded')==='true'){closeMenu();menuButton.focus();}});
const figureDialog=byId('figure-dialog');
document.querySelectorAll('[data-figure]').forEach(button=>button.addEventListener('click',()=>{byId('figure-image').src=button.dataset.figure;byId('figure-title').textContent=tr('MEDEA training framework','MEDEA 训练框架');figureDialog.showModal();document.body.classList.add('no-scroll');}));
byId('close-figure').addEventListener('click',()=>figureDialog.close());
figureDialog.addEventListener('click',e=>{if(e.target===figureDialog){const rect=figureDialog.getBoundingClientRect();if(e.clientX<rect.left||e.clientX>rect.right||e.clientY<rect.top||e.clientY>rect.bottom)figureDialog.close();}});
figureDialog.addEventListener('close',()=>document.body.classList.remove('no-scroll'));
byId('copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(byId('bibtex').textContent);byId('copy-status').textContent=tr('Citation copied. Thank you for building on this work.','引用已复制。感谢你在这项研究上继续探索。');}catch{const range=document.createRange();range.selectNodeContents(byId('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);byId('copy-status').textContent=tr('Citation selected. Press Ctrl/Cmd+C, or download the BibTeX file.','引用已选中。按 Ctrl/Cmd+C 复制，或下载 BibTeX 文件。');}});
let savedLanguage;try{savedLanguage=localStorage.getItem('caster-language');}catch{}
setLanguage(savedLanguage==='zh'?'zh':'en');
