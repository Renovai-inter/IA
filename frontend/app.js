/**
 * Renovai IA — Frontend Funcional
 * Conexão direta com a API FastAPI (POST /chat e POST /sessions/{id}/encerrar).
 * Baseado na arquitetura e boas práticas de integração do migracao_fastAPI.
 */

// ============================================================
// Configuração de API
// ============================================================
const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:8000' : '';
const CHAT_ENDPOINT = `${API_BASE}/chat`;
const HEALTH_ENDPOINT = `${API_BASE}/health`;
const SESSIONS_ENDPOINT = (id) => `${API_BASE}/sessions/${encodeURIComponent(id)}/encerrar`;

// ============================================================
// Chaves de Armazenamento Local
// ============================================================
const KEY_SESSION = 'renovai_session_id';
const KEY_DOMINIO = 'renovai_dominio_tipo';
const KEY_COOP_ID = 'renovai_cooperativa_id';
const KEY_EMP_ID  = 'renovai_empresa_id';
const KEY_USER_ID = 'renovai_user_id';

// Defaults
const DEFAULT_COOP_ID = 'cooperativa-central-sp';
const DEFAULT_EMP_ID  = 'empresa-sustentavel-sp';
const DEFAULT_USER_ID = 'usuario_renovai';

// Poses do Nimo (frontend/assets)
const NIMO = {
  avatar:   './assets/cabeca_nimo.png',
  feliz:    './assets/nimo_feliz.png',
  pensando: './assets/nimo_pensando.png',
  triste:   './assets/nimo_triste.png',
};

// Logos por domínio (Cooperativa = verde · Empresa = vinho)
const LOGOS = {
  COOPERATIVA: './assets/logo_renovai_cooperativa.svg',
  EMPRESA:     './assets/logo_renovai_empresa.svg',
};

const TEXTO_BOAS_VINDAS = {
  COOPERATIVA: 'Tire dúvidas sobre triagem de materiais, legislação PNRS, consulte estoque e otimize rotas de coleta. Eu chamo os agentes certos para ajudar você.',
  EMPRESA:     'Encontre cooperativas parceiras, acompanhe os resíduos gerados pela sua empresa, entenda a logística reversa e agende coletas. Eu chamo os agentes certos para ajudar você.',
};

// Sugestões rápidas por domínio
const SUGESTOES = {
  COOPERATIVA: [
    { rotulo: 'Como separar plásticos?',  prompt: 'Como separar resíduos plásticos corretamente para a reciclagem?' },
    { rotulo: 'Consultar estoque',        prompt: 'Quanto de PET e papelão temos registrado no estoque atual?' },
    { rotulo: 'Otimizar rota de coleta',  prompt: 'Otimizar rota de coleta para amanhã na região central de São Paulo' },
    { rotulo: 'Legislação PNRS',          prompt: 'O que diz a PNRS sobre responsabilidade compartilhada e logística reversa?' },
  ],
  EMPRESA: [
    { rotulo: 'Encontrar cooperativas',   prompt: 'Quais cooperativas parceiras possuem capacidade para entrega de recicláveis?' },
    { rotulo: 'Meus resíduos gerados',    prompt: 'Quanto resíduo reciclável a minha empresa gerou neste mês?' },
    { rotulo: 'Logística reversa',        prompt: 'O que diz a PNRS sobre responsabilidade compartilhada e logística reversa?' },
    { rotulo: 'Agendar coleta',           prompt: 'Agendar coleta de recicláveis para a próxima semana' },
  ],
};

const TEXTO_DIGITANDO = 'O Nimo está mexendo as folhinhas para analisar os dados... 🌿🐾';

// ============================================================
// Elementos de UI
// ============================================================
const elements = {
  messages:        document.getElementById('messages'),
  welcomeCard:     document.getElementById('welcome-card'),
  chatForm:        document.getElementById('chatForm'),
  messageInput:    document.getElementById('messageInput'),
  sendButton:      document.getElementById('sendButton'),
  statusDot:       document.getElementById('status-dot'),
  statusText:      document.getElementById('status-text'),
  sessionIdEl:     document.getElementById('session-id'),
  resetSessionBtn: document.getElementById('reset-session'),
  statusHint:      document.getElementById('statusHint'),
  domainBadge:     document.getElementById('domain-badge'),
  btnCooperativa:  document.getElementById('btn-cooperativa'),
  btnEmpresa:      document.getElementById('btn-empresa'),
  brandLogo:       document.getElementById('brand-logo'),
  headerState:     document.getElementById('header-state'),
  suggestions:     document.getElementById('suggestions'),
  welcomeText:     document.getElementById('welcome-text'),
};

// ============================================================
// Gestão de Sessão & Domínio
// ============================================================
function gerarSessionId() {
  if (window.crypto && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return 'sess-' + Math.random().toString(36).slice(2, 11);
}

function obterOuCriarSessionId() {
  let id = localStorage.getItem(KEY_SESSION);
  if (!id) {
    id = gerarSessionId();
    localStorage.setItem(KEY_SESSION, id);
  }
  return id;
}

function exibirSessionId(id) {
  if (elements.sessionIdEl) {
    elements.sessionIdEl.textContent = id.length > 12 ? id.slice(0, 8) + '…' : id;
    elements.sessionIdEl.title = `Sessão ativa: ${id}`;
  }
}

let sessionId = obterOuCriarSessionId();
exibirSessionId(sessionId);

let dominioAtual = localStorage.getItem(KEY_DOMINIO) || 'COOPERATIVA';

function renderSugestoes() {
  if (!elements.suggestions) return;
  elements.suggestions.innerHTML = '';
  (SUGESTOES[dominioAtual] || SUGESTOES.COOPERATIVA).forEach((item) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'suggestion';
    btn.dataset.prompt = item.prompt;
    btn.textContent = item.rotulo;
    elements.suggestions.appendChild(btn);
  });
}

function atualizarSeletorDominio() {
  const isCoop = dominioAtual === 'COOPERATIVA';
  if (elements.btnCooperativa) {
    elements.btnCooperativa.classList.toggle('is-active', isCoop);
    elements.btnCooperativa.setAttribute('aria-pressed', String(isCoop));
  }
  if (elements.btnEmpresa) {
    elements.btnEmpresa.classList.toggle('is-active', !isCoop);
    elements.btnEmpresa.setAttribute('aria-pressed', String(!isCoop));
  }

  // Tema visual: verde (Cooperativa) ou vinho (Empresa)
  document.documentElement.setAttribute('data-tema', dominioAtual);
  if (elements.brandLogo) elements.brandLogo.src = LOGOS[dominioAtual] || LOGOS.COOPERATIVA;
  if (elements.welcomeText) elements.welcomeText.textContent = TEXTO_BOAS_VINDAS[dominioAtual] || TEXTO_BOAS_VINDAS.COOPERATIVA;
  renderSugestoes();

  const idAlvo = isCoop
    ? (localStorage.getItem(KEY_COOP_ID) || DEFAULT_COOP_ID)
    : (localStorage.getItem(KEY_EMP_ID) || DEFAULT_EMP_ID);

  if (elements.domainBadge) {
    elements.domainBadge.textContent = `${isCoop ? 'Cooperativa' : 'Empresa'}: ${idAlvo}`;
    elements.domainBadge.title = `Contexto ativo: ${dominioAtual} (${idAlvo})`;
  }
}

function definirDominio(tipo) {
  dominioAtual = tipo;
  localStorage.setItem(KEY_DOMINIO, tipo);
  atualizarSeletorDominio();
  setHint(`Contexto alterado para ${tipo === 'COOPERATIVA' ? 'Cooperativa' : 'Empresa'}.`);
}

if (elements.btnCooperativa) {
  elements.btnCooperativa.addEventListener('click', () => definirDominio('COOPERATIVA'));
}
if (elements.btnEmpresa) {
  elements.btnEmpresa.addEventListener('click', () => definirDominio('EMPRESA'));
}
atualizarSeletorDominio();

// ============================================================
// Encerrar e Reiniciar Sessão
// ============================================================
async function encerrarSessaoServidor(id) {
  try {
    const resposta = await fetch(SESSIONS_ENDPOINT(id), { method: 'POST' });
    if (!resposta.ok) return null;
    const dados = await resposta.json();
    return dados.resumo || null;
  } catch (erro) {
    console.warn('Não foi possível encerrar a sessão no servidor:', erro);
    return null;
  }
}

async function reiniciarSessao() {
  const anterior = sessionId;
  elements.resetSessionBtn.disabled = true;
  marcarStatus('busy', 'Encerrando sessão...');
  setHint('Resumindo a conversa e gravando na memória de longo prazo...');

  const resumo = await encerrarSessaoServidor(anterior);

  // Gera e persiste o novo session_id
  const novoId = gerarSessionId();
  localStorage.setItem(KEY_SESSION, novoId);
  sessionId = novoId;
  exibirSessionId(novoId);

  // Limpa mensagens e restaura o card de boas-vindas
  elements.messages.innerHTML = '';
  if (elements.welcomeCard) {
    elements.messages.appendChild(elements.welcomeCard);
    elements.welcomeCard.style.display = 'block';
  }

  marcarStatus('online', 'online e pronto');
  setHint(
    resumo
      ? `Sessão anterior encerrada e resumida na memória: "${resumo.slice(0, 80)}${resumo.length > 80 ? '…' : ''}"`
      : 'Nova sessão iniciada com sucesso.',
    false,
    true
  );

  elements.resetSessionBtn.disabled = false;
  elements.messageInput.focus();
}

if (elements.resetSessionBtn) {
  elements.resetSessionBtn.addEventListener('click', reiniciarSessao);
}

// ============================================================
// Utilidades de UI & Formatação
// ============================================================
function setHint(texto, comoErro = false, comoSucesso = false) {
  if (!elements.statusHint) return;
  elements.statusHint.textContent = texto;
  elements.statusHint.classList.toggle('is-error', comoErro);
  elements.statusHint.classList.toggle('is-success', comoSucesso);
}

function marcarEstadoNimo(texto) {
  if (elements.headerState) elements.headerState.textContent = texto;
}

function marcarStatus(estado, textoStatus) {
  if (elements.statusDot) {
    elements.statusDot.classList.toggle('is-offline', estado === 'offline');
    elements.statusDot.classList.toggle('is-busy', estado === 'busy');
  }
  if (elements.statusText) {
    elements.statusText.textContent = textoStatus || (estado === 'online' ? 'online e pronto' : 'offline');
  }
}

function escaparHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatarInline(html) {
  // Código inline `codigo`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Negrito **texto**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Itálico *texto*
  html = html.replace(/(^|[^*])\*(?!\s)([^*]+?)\*(?!\*)/g, '$1<em>$2</em>');
  return html;
}

function formatarMarkdown(texto) {
  if (!texto) return '';

  // 1) Isola blocos de código ``` ``` antes de escapar o restante
  const blocos = [];
  let bruto = String(texto).replace(/```[a-zA-Z0-9_-]*\n?([\s\S]*?)```/g, (_, codigo) => {
    blocos.push(`<pre><code>${escaparHtml(codigo.replace(/\n$/, ''))}</code></pre>`);
    return `\u0000${blocos.length - 1}\u0000`;
  });

  // 2) Escapa HTML e processa linha a linha (parágrafos, títulos e listas)
  const linhas = escaparHtml(bruto).split('\n');
  const saida = [];
  let lista = null;      // 'ul' | 'ol'
  let paragrafo = [];

  const fecharParagrafo = () => {
    if (paragrafo.length) {
      saida.push(`<p>${paragrafo.map(formatarInline).join('<br>')}</p>`);
      paragrafo = [];
    }
  };
  const fecharLista = () => {
    if (lista) { saida.push(`</${lista}>`); lista = null; }
  };

  linhas.forEach((linha) => {
    const t = linha.trim();
    let m;
    if (!t) { fecharParagrafo(); fecharLista(); return; }
    if ((m = t.match(/^\u0000(\d+)\u0000$/))) {
      fecharParagrafo(); fecharLista();
      saida.push(blocos[Number(m[1])]);
    } else if ((m = t.match(/^(#{1,4})\s+(.*)$/))) {
      fecharParagrafo(); fecharLista();
      const nivel = m[1].length <= 2 ? 'h3' : 'h4';
      saida.push(`<${nivel}>${formatarInline(m[2])}</${nivel}>`);
    } else if ((m = t.match(/^[-*•]\s+(.*)$/))) {
      fecharParagrafo();
      if (lista !== 'ul') { fecharLista(); saida.push('<ul>'); lista = 'ul'; }
      saida.push(`<li>${formatarInline(m[1])}</li>`);
    } else if ((m = t.match(/^\d+[.)]\s+(.*)$/))) {
      fecharParagrafo();
      if (lista !== 'ol') { fecharLista(); saida.push('<ol>'); lista = 'ol'; }
      saida.push(`<li>${formatarInline(m[1])}</li>`);
    } else {
      fecharLista();
      paragrafo.push(t);
    }
  });
  fecharParagrafo();
  fecharLista();

  return saida.join('');
}

function rolarParaFinal() {
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function ocultarWelcome() {
  if (elements.welcomeCard && elements.welcomeCard.style.display !== 'none') {
    elements.welcomeCard.style.display = 'none';
  }
}

// ============================================================
// Inserção de Mensagens
// ============================================================
function adicionarMensagemUsuario(texto) {
  ocultarWelcome();
  const row = document.createElement('div');
  row.className = 'user-row';

  const bubble = document.createElement('div');
  bubble.className = 'user-bubble';
  bubble.textContent = texto;

  row.appendChild(bubble);
  elements.messages.appendChild(row);
  rolarParaFinal();
}

function adicionarMensagemAssistente({ texto, agentes, tipo = 'assistant' }) {
  const msg = document.createElement('div');
  msg.className = `message message--${tipo}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  const img = document.createElement('img');
  img.src = tipo === 'error' ? NIMO.triste : NIMO.avatar;
  img.alt = tipo === 'error' ? 'Nimo triste' : 'Nimo, mascote do Renovaí';
  avatar.appendChild(img);
  msg.appendChild(avatar);

  const content = document.createElement('div');
  content.className = 'message-content';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = formatarMarkdown(texto);
  content.appendChild(bubble);

  // Tags para agentes especialistas chamados no turno (mesmo conceito do migracao_fastAPI)
  if (agentes && Array.isArray(agentes) && agentes.length > 0) {
    const agentesContainer = document.createElement('div');
    agentesContainer.className = 'message__agents';
    agentes.forEach((nome) => {
      if (nome && nome !== '__reset_agentes__') {
        const tag = document.createElement('span');
        tag.className = 'agent-tag';
        tag.textContent = nome;
        agentesContainer.appendChild(tag);
      }
    });
    if (agentesContainer.children.length > 0) {
      content.appendChild(agentesContainer);
    }
  }

  msg.appendChild(content);
  elements.messages.appendChild(msg);
  rolarParaFinal();
  return msg;
}

function adicionarIndicadorDigitando() {
  const msg = document.createElement('div');
  msg.className = 'message message--assistant';
  msg.id = 'typing-indicator';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  const img = document.createElement('img');
  img.src = NIMO.pensando;
  img.alt = 'Nimo pensando';
  avatar.appendChild(img);
  msg.appendChild(avatar);

  const content = document.createElement('div');
  content.className = 'message-content';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>'
    + `<div class="typing-label">${TEXTO_DIGITANDO}</div>`;
  content.appendChild(bubble);

  msg.appendChild(content);
  elements.messages.appendChild(msg);
  rolarParaFinal();
}

function removerIndicadorDigitando() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

// ============================================================
// Auto-resize do Textarea
// ============================================================
function ajustarAlturaTextarea() {
  const input = elements.messageInput;
  if (!input) return;
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 130) + 'px';
  elements.sendButton.disabled = !input.value.trim();
}

elements.messageInput.addEventListener('input', ajustarAlturaTextarea);

elements.messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

// ============================================================
// Pílulas de Sugestão
// ============================================================
if (elements.suggestions) {
  elements.suggestions.addEventListener('click', (e) => {
    const btn = e.target.closest('.suggestion');
    if (!btn || elements.messageInput.disabled) return;
    elements.messageInput.value = btn.dataset.prompt || btn.textContent.trim();
    ajustarAlturaTextarea();
    elements.chatForm.requestSubmit();
  });
}

// ============================================================
// Envio do Chat (POST /chat)
// ============================================================
elements.chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const pergunta = elements.messageInput.value.trim();
  if (!pergunta) return;

  adicionarMensagemUsuario(pergunta);
  elements.messageInput.value = '';
  ajustarAlturaTextarea();
  setHint('');

  elements.sendButton.disabled = true;
  elements.messageInput.disabled = true;
  marcarStatus('busy', 'Processando resposta...');
  marcarEstadoNimo('pensando...');
  adicionarIndicadorDigitando();

  // Monta payload respeitando o validador de domínio do ChatRequest
  const isCoop = dominioAtual === 'COOPERATIVA';
  const payload = {
    session_id: sessionId,
    pergunta: pergunta,
    user_id: localStorage.getItem(KEY_USER_ID) || DEFAULT_USER_ID,
  };

  if (isCoop) {
    payload.cooperativa_id = localStorage.getItem(KEY_COOP_ID) || DEFAULT_COOP_ID;
    payload.empresa_id = null;
  } else {
    payload.empresa_id = localStorage.getItem(KEY_EMP_ID) || DEFAULT_EMP_ID;
    payload.cooperativa_id = null;
  }

  try {
    const resposta = await fetch(CHAT_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!resposta.ok) {
      let detalhe = '';
      try {
        const erroJson = await resposta.json();
        detalhe = erroJson.detail || JSON.stringify(erroJson);
      } catch {
        detalhe = await resposta.text();
      }

      if (resposta.status === 429) {
        throw new Error(`Limite de requisições excedido. ${detalhe || 'Por favor, aguarde alguns segundos antes de tentar novamente.'}`);
      }

      throw new Error(`A API respondeu com status ${resposta.status}: ${detalhe || 'Sem detalhes no corpo do erro.'}`);
    }

    const dados = await resposta.json();
    marcarStatus('online', 'online e pronto');
    removerIndicadorDigitando();

    adicionarMensagemAssistente({
      texto: dados.resposta || '(Sem resposta da IA)',
      agentes: dados.agentes_chamados || [],
    });
    setHint('Resposta concluída.', false, true);

  } catch (erro) {
    marcarStatus('offline', 'erro de conexão');
    removerIndicadorDigitando();

    const mensagemAmigavel = erro.message.includes('Failed to fetch')
      ? `Não foi possível conectar ao servidor em ${API_BASE || 'localhost:8000'}. Verifique se o backend do Renovai IA está ativo.`
      : erro.message;

    adicionarMensagemAssistente({
      texto: `Desculpe, ocorreu um problema ao processar sua solicitação:\n\n**${mensagemAmigavel}**`,
      tipo: 'error',
    });
    setHint(mensagemAmigavel, true);
    console.error('Erro na chamada /chat:', erro);

  } finally {
    marcarEstadoNimo('online e pronto');
    elements.sendButton.disabled = false;
    elements.messageInput.disabled = false;
    ajustarAlturaTextarea();
    elements.messageInput.focus();
  }
});

// ============================================================
// Verificação Inicial de Conexão (Healthcheck)
// ============================================================
async function verificarSaudeBackend() {
  try {
    const res = await fetch(HEALTH_ENDPOINT);
    if (res.ok) {
      marcarStatus('online', 'online e pronto');
      setHint('Conectado à API do Renovai IA.');
    } else {
      marcarStatus('busy', 'iniciando');
    }
  } catch {
    marcarStatus('offline', 'offline');
    setHint('Servidor desconectado. Certifique-se de que a API está rodando.', true);
  }
}

window.addEventListener('load', () => {
  rolarParaFinal();
  verificarSaudeBackend();
  ajustarAlturaTextarea();
  elements.messageInput.focus();
});
