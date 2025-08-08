# backend/app/models.py

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime

# Cria a "fábrica" de classes Base para o ORM
Base = declarative_base()

# Define a classe Game, que mapeia para a tabela games_dim do SQLite
class Game(Base):
    __tablename__ = "games_dim"   # Nome exato da tabela no banco

    # Coluna de chave primária 'game_id'
    game_id = Column(
        Integer,      # tipo inteiro
        primary_key=True,  # é a primary key
        index=True        # cria um índice para buscas mais rápidas
    )

    # Coluna 'slug' (string única, não-nula)
    slug = Column(
        String,       # tipo texto
        unique=True,  # não pode haver dois jogos com o mesmo slug
        nullable=False  # valor obrigatório
    )

    # Colunas de texto para outras informações
    title = Column(String)        # título original do jogo
    genres = Column(String)       # gênero(s), ex: "Action,RPG"
    Platform = Column(String)     # plataforma, ex: "PS4"
    Publisher = Column(String)    # nome da publisher
    Year = Column(Integer)        # ano de lançamento

    # Colunas numéricas para vendas e notas
    Global_Sales = Column(Float)  # vendas globais (milhões)
    metascore = Column(Float)     # nota crítica média
    userscore = Column(Float)     # nota de usuários média

    # 7)  Coluna de data original
    releaseDate = Column(DateTime)  # data completa de lançamento
