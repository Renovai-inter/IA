from __future__ import annotations

from app.a2a.schemas import AgentCapabilities, AgentCard, AgentSkill

ROTAS_SERVICO_COOPERATIVA = ('estoque', 'coleta', 'financeiro', 'logistica')
ROTAS_SERVICO_EMPRESA = ('pedidos',)


def get_cooperativa_agent_card(base_url: str = 'http://localhost:8000/a2a/cooperativa') -> AgentCard:
    """Retorna o Agent Card (Google A2A) do Microsserviço de Cooperativas (/a2a/cooperativa)."""
    return AgentCard(
        name='Renovaí Cooperativa Agent Service',
        description=(
            'Microsserviço A2A responsável pelo subgrafo de especialistas operacionais de '
            'cooperativas de reciclagem (estoque, coleta/triagem, financeiro/rateio e logística), '
            'operando diretamente sobre o cooperativa_id da requisição.'
        ),
        url=base_url.rstrip('/'),
        version='1.0.0',
        protocolVersion='0.2.1',
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=True,
        ),
        skills=[
            AgentSkill(
                id='estoque',
                name='Gestão de Materiais e Estoque (MaterialEstoqueAgent)',
                description=(
                    'Consulta saldos atuais em galpão, categorias de materiais recicláveis e '
                    'histórico de movimentações de entrada e saída filtrados por cooperativa_id.'
                ),
                tags=['estoque', 'materiais', 'lotes', 'cooperativa'],
                examples=[
                    'Quanto de papelão e PET temos em estoque hoje?',
                    'Qual foi a movimentação de entrada desta semana?',
                ],
            ),
            AgentSkill(
                id='coleta',
                name='Gestão de Coleta e Triagem (ColetaAgent)',
                description=(
                    'Consulta coletas agendadas/concluídas, triagens, equipes operacionais e '
                    'taxa de rejeito da cooperativa filtrados por cooperativa_id.'
                ),
                tags=['coleta', 'triagem', 'equipes', 'rejeito', 'cooperativa'],
                examples=[
                    'Quantas triagens foram realizadas hoje e qual a taxa de rejeito?',
                ],
            ),
            AgentSkill(
                id='financeiro',
                name='Gestão Financeira e Rateio (FinanceiroAgent)',
                description=(
                    'Consulta rateios mensais, repasses individuais de cooperados e despesas '
                    'operacionais da cooperativa filtrados por cooperativa_id.'
                ),
                tags=['financeiro', 'rateio', 'despesas', 'cooperados', 'cooperativa'],
                examples=[
                    'Qual o valor líquido do rateio deste mês?',
                ],
            ),
            AgentSkill(
                id='logistica',
                name='Otimização Supervisionada de Rotas (LogisticaAgent)',
                description=(
                    'Calcula e persiste no Neo4j a sequência ótima de visitas de coleta em '
                    'São Paulo (GradientBoosting + 2-Opt) para o cooperativa_id.'
                ),
                tags=['logistica', 'rotas', 'otimizacao', 'cooperativa'],
                examples=[
                    'Qual a melhor rota saindo do Tatuapé às 8h passando pela Mooca e Pinheiros?',
                ],
            ),
        ],
    )


def get_empresa_agent_card(base_url: str = 'http://localhost:8000/a2a/empresa') -> AgentCard:
    """Retorna o Agent Card (Google A2A) do Microsserviço de Empresas Compradoras (/a2a/empresa)."""
    return AgentCard(
        name='Renovaí Empresa Agent Service',
        description=(
            'Microsserviço A2A responsável pelo subgrafo de especialistas de empresas compradoras '
            '(pedidos, sourcing e recomendação multicritério de cooperativas no Neo4j), '
            'operando diretamente sobre o empresa_id da requisição.'
        ),
        url=base_url.rstrip('/'),
        version='1.0.0',
        protocolVersion='0.2.1',
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=True,
        ),
        skills=[
            AgentSkill(
                id='pedidos',
                name='Recomendação e Sourcing de Cooperativas (RecomendacaoAgent)',
                description=(
                    'Ranqueia cooperativas fornecedoras no grafo Neo4j com base no empresa_id, '
                    'distância na malha viária, estoque disponível, capacidade mensal e conformidade.'
                ),
                tags=['pedidos', 'recomendacao', 'sourcing', 'empresa', 'neo4j'],
                examples=[
                    'Me dê um ranking de cooperativas num raio de 30 km para comprar 1000 kg de PET',
                ],
            ),
        ],
    )
