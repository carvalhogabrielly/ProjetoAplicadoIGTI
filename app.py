import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class Recomendador:
    def __init__(self, modelo_path='modelo_knn.pkl', dados_path='book_pivot.pkl'):
        self.modelo = self.carregar_modelo(modelo_path)
        self.book_pivot = self.carregar_dados(dados_path)

    @staticmethod
    def carregar_modelo(path):
        return joblib.load(path)

    @staticmethod
    def carregar_dados(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def recomendar(self, book_name, n_neighbors=5):
        try:
            book_id = np.where(self.book_pivot.index == book_name)[0][0]
            distances, suggestions = self.modelo.kneighbors(
                self.book_pivot.iloc[book_id, :].values.reshape(1, -1), 
                n_neighbors=n_neighbors + 1
            )
            # Excluir o próprio livro da recomendação
            books = [self.book_pivot.index[s] for s in suggestions[0] if s != book_id]
            return books
        except IndexError:
            st.error("Livro não encontrado no sistema!")
            return []

class AplicativoStreamlit:
    def __init__(self, recomendador):
        self.recomendador = recomendador

    def run(self):
        st.title("Recomendação de Livros 📚")

        with st.sidebar:
            st.header("Perfil do Usuário")
            nome = st.text_input("Digite seu nome:")
            if nome:
                if nome not in st.session_state:
                    st.session_state[nome] = {"favoritos": []}
                st.write(f"Olá, {nome}!")

        if nome:
            self.exibir_favoritos(nome)
            self.exibir_recomendacoes(nome)
            self.coletar_feedback()

    def exibir_favoritos(self, nome):
        favoritos = st.session_state[nome]["favoritos"]

        novo_livro = st.selectbox(
            "Adicione um livro aos favoritos:",
            [livro for livro in self.recomendador.book_pivot.index if livro not in favoritos]
        )
        if st.button("Adicionar Livro"):
            favoritos.append(novo_livro)
            st.success(f"'{novo_livro}' adicionado aos favoritos!")

        if favoritos:
            st.write("Seus livros favoritos:")
            for livro in favoritos:
                if st.button(f"Remover {livro}", key=livro):
                    favoritos.remove(livro)
                    st.success(f"'{livro}' removido dos favoritos!")

    def exibir_recomendacoes(self, nome):
        favoritos = st.session_state[nome]["favoritos"]
        if favoritos:
            ultimo_livro = favoritos[-1]
            recomendacoes = self.recomendador.recomendar(ultimo_livro)
            st.subheader(f"Recomendações com base em '{ultimo_livro}':")
            for rec in recomendacoes:
                st.write(f"- {rec}")

    def coletar_feedback(self):
        st.subheader("Feedback das Recomendações")
        feedback = st.radio("Você gostou das recomendações?", ("Sim", "Não"))
        comentario = st.text_area("Comentários adicionais (opcional):")
        if st.button("Enviar Feedback"):
            st.success("Obrigado pelo seu feedback!")


def main():
    recomendador = Recomendador()
    app = AplicativoStreamlit(recomendador)
    app.run()

if __name__ == "__main__":
    main()
