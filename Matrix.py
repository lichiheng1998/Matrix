from abc import ABC, abstractmethod
from my_matrix.Mapping import map_2d


class DimensionNotMatchError(Exception):
    pass


class AbstractMatrix(ABC):
    """ A class which represent the matrix """

    def __init__(self, row=0, col=0, content=None):
        self.set_content(self.container())
        self.set_dimension(row, col)
        if content is None:
            self.set_content(map_2d(lambda: 0, row, col,
                             iter_key=[self.iter],
                             assign_key=self.assign,
                             construct=self.container()))
        else:
            self._init_by_content(content)

    def _init_by_content(self, content: list):
        row = len(content)
        col = len(content[0])
        self.set_dimension(row, col)
        self.set_content(map_2d(lambda x: x, row, col, content,
                         iter_key=[lambda item, r, c: item[r][c]],
                         assign_key=self.assign,
                         construct=self.container()
                         ))

    def transpose(self: 'AbstractMatrix') \
            -> 'AbstractMatrix':
        """ Return the transpose of the current matrix """
        (row, col) = (self.get_row(), self.get_col())
        matrix = self.instance(col, row)
        content = map_2d(lambda x: x, row, col, self.get_content(),
                         construct=self.container(),
                         iter_key=[self.iter],
                         assign_key=self.assign_inverse
                         )
        matrix.set_content(content)
        return matrix

    def assign_inverse(self, container, r, c, value):
        self.assign(container, c, r, value)

    @staticmethod
    @abstractmethod
    def container():
        pass

    @staticmethod
    @abstractmethod
    def assign(container, r, c, value):
        pass

    @staticmethod
    @abstractmethod
    def iter(container, r, c):
        pass

    @staticmethod
    @abstractmethod
    def instance(r, c) -> 'AbstractMatrix':
        pass

    @abstractmethod
    def dot(self: 'AbstractMatrix', matrix: 'AbstractMatrix')\
            -> 'AbstractMatrix':
        """ Let the current matrix to be A, Given another matrix B,
        calculate and return A dot B """
        pass

    # def multiply(self: 'AbstractMatrix', matrix: 'AbstractMatrix'):

    @abstractmethod
    def cross(self: 'AbstractMatrix', matrix: 'AbstractMatrix') \
            -> 'AbstractMatrix':
        """ Let the current matrix to be A, Given another matrix B,
        calculate and return A cross B """
        pass

    def _cal(self, func, matrix=None) -> 'AbstractMatrix':
        row = self.get_row()
        col = self.get_col()
        new_matrix = self.instance(row, col)
        args = [self.get_content()]
        iters = [self.iter]
        if matrix is not None:
            if not self.check_dimension_equality(matrix):
                raise DimensionNotMatchError('Dimension Not match!')
            args.append(matrix.get_content())
            iters.append(matrix.iter)
        content = map_2d(func, row, col, *args,
                         iter_key=iters,
                         assign_key=self.assign,
                         construct=self.container())
        new_matrix.set_content(content)
        return new_matrix

    def add(self: 'AbstractMatrix', matrix: 'AbstractMatrix') \
            -> 'AbstractMatrix':
        """ Let the current matrix to be A, Given another matrix B,
        calculate and return A + B """
        return self._cal(lambda x, y: x + y, matrix)

    def minus(self: 'AbstractMatrix', matrix: 'AbstractMatrix') \
            -> 'AbstractMatrix':
        """ Let the current matrix to be A, Given another matrix B,
        calculate and return A - B """
        return self._cal(lambda x, y: x - y, matrix)

    def scalar_mul(self: 'AbstractMatrix', scalar: int or float) \
            -> 'AbstractMatrix':
        """ Let the current matrix to be A, Given another scalar c,
        calculate and return cA """
        return self._cal(lambda x: scalar * x)

    def __add__(self, other: 'AbstractMatrix') -> 'AbstractMatrix':
        return self.add(other)

    def __sub__(self, other: 'AbstractMatrix') -> 'AbstractMatrix':
        return self.minus(other)

    def __mul__(self, other: int or float) -> 'AbstractMatrix':
        return self.scalar_mul(other)

    def __rmul__(self, other: int or float) -> 'AbstractMatrix':
        return self.scalar_mul(other)

    def __eq__(self, other: 'AbstractMatrix') -> bool:
        return self.is_equal(other)

    def __str__(self):
        matrix_str = ''
        for r in range(self.get_row()):
            for c in range(self.get_col()):
                matrix_str += str(self.getitem(r, c)) + '\t'
            matrix_str += '\n'

        return matrix_str

    @abstractmethod
    def getitem(self: 'AbstractMatrix', row: int, col: int):
        """ Get the item from the matrix using the row and the col """
        pass

    @abstractmethod
    def setitem(self: 'AbstractMatrix', row: int, col: int, item):
        """ Set the item to the matrix using the row and the col """
        pass

    def get_by_col(self: 'AbstractMatrix', col: int) -> 'AbstractMatrix':
        """ Get the col vector by the given col number """
        row = self.get_row()
        col_vector = Matrix(row, 1)
        for row in range(row):
            col_vector.setitem(row, 0, self.getitem(row, col))
        return col_vector

    def get_by_row(self: 'AbstractMatrix', row: int) -> 'AbstractMatrix':
        """ Get the row vector by the given row number """
        col = self.get_col()
        row_vector = Matrix(1, col)
        for col in range(col):
            row_vector.setitem(0, col, self.getitem(row, col))
        return row_vector

    def is_equal(self: 'AbstractMatrix', matrix: 'AbstractMatrix') -> bool:
        """ Return whether two matrix are equal """
        row = self.get_row()
        col = self.get_col()
        return (self.check_dimension_equality(matrix) and
                self._check_equality(matrix, row, col))

    def check_dimension_equality(self: 'AbstractMatrix', m2: 'AbstractMatrix')\
            -> bool:
        """ Return whether two matrix have the same dimension """
        return self.get_row() == m2.get_row() and self.get_col() == m2.get_col()

    def _check_equality(self, m2, row, col, r=0, c=0):
        if self.getitem(r, c) != m2.getitem(r, c):
            return False
        elif r == row - 1 and c == col - 1:
            return True
        (r, c) = (r + 1, 0) if c == col - 1 else (r, c + 1)
        return self._check_equality(m2, row, col, r, c)

    @abstractmethod
    def get_row(self: 'AbstractMatrix') -> int:
        """ Return the row number """
        pass

    @abstractmethod
    def get_col(self: 'AbstractMatrix') -> int:
        """ Return the col number """
        pass

    @abstractmethod
    def set_dimension(self: 'AbstractMatrix', row: int, col: int):
        pass

    @abstractmethod
    def get_content(self: 'AbstractMatrix'):
        """ Return the row number """
        pass

    @abstractmethod
    def set_content(self: 'AbstractMatrix', content):
        """ Return the col number """
        pass


class ListMatrix(AbstractMatrix):

    def __init__(self, row=0, col=0, content=None):
        self.row_len = None
        self.col_len = None
        self._content = None
        super().__init__(row, col, content)

    def dot(self: 'ListMatrix', matrix: 'ListMatrix') \
            -> 'ListMatrix':
        """ Let the current matrix to be A, Given another matrix B,
        calculate and return A dot B """
        pass

    def cross(self: 'ListMatrix', matrix: 'ListMatrix') \
            -> 'ListMatrix':
        """ Let the current matrix to be A, Given another matrix B,
        calculate and return A cross B """
        pass

    def getitem(self: 'ListMatrix', row: int, col: int):
        """ Get the item from the matrix using the row and the col """
        return self._content[row][col]

    def setitem(self: 'ListMatrix', row: int, col: int, item) -> None:
        """ Set the item to the matrix using the row and the col """
        self._content[row][col] = item

    @staticmethod
    def assign(container, r, c, value):
        while len(container) < r + 1:
            container.append([])
        container[r].append(value)

    @staticmethod
    def iter(container, r, c):
        return container[r][c]

    @staticmethod
    def instance(r, c) -> 'ListMatrix':
        return ListMatrix(r, c)

    def get_row(self: 'ListMatrix') -> int:
        return self.row_len

    def get_col(self: 'ListMatrix') -> int:
        """ Return the col number """
        return self.col_len

    def set_dimension(self: 'ListMatrix', row: int, col: int):
        (self.row_len, self.col_len) = (row, col)

    def get_content(self: 'ListMatrix'):
        """ Return the row number """
        return self._content

    def set_content(self: 'ListMatrix', content):
        """ Return the col number """
        self._content = content

    @staticmethod
    def container():
        return []


class Matrix(AbstractMatrix):

    def __init__(self, row=0, col=0, content=None):
        self.row_len = None
        self.col_len = None
        self._content = None
        super().__init__(row, col, content)

    def dot(self: 'AbstractMatrix', matrix: 'AbstractMatrix') \
            -> 'AbstractMatrix':
        """ Let the current matrix to be A, Given another matrix B,
        calculate and return A dot B """
        pass

    def cross(self: 'AbstractMatrix', matrix: 'AbstractMatrix') \
            -> 'AbstractMatrix':
        """ Let the current matrix to be A, Given another matrix B,
        calculate and return A cross B """
        pass

    def getitem(self: 'Matrix', row: int, col: int):
        """ Get the item from the matrix using the row and the col """
        return self._content[(row, col)]

    def setitem(self: 'Matrix', row: int, col: int, item) -> None:
        """ Set the item to the matrix using the row and the col """
        self._content[(row, col)] = item

    @staticmethod
    def assign(container, r, c, value):
        container[(r, c)] = value

    @staticmethod
    def iter(container, r, c):
        return container[(r, c)]

    @staticmethod
    def instance(r, c) -> 'AbstractMatrix':
        return Matrix(r, c)

    def get_row(self: 'Matrix') -> int:
        return self.row_len

    def get_col(self: 'Matrix') -> int:
        """ Return the col number """
        return self.col_len

    def set_dimension(self: 'Matrix', row: int, col: int):
        (self.row_len, self.col_len) = (row, col)

    def get_content(self: 'Matrix'):
        """ Return the row number """
        return self._content

    def set_content(self: 'Matrix', content):
        """ Return the col number """
        self._content = content

    @staticmethod
    def container():
        return{}


a = ListMatrix(content=[[1, 2, 5], [3, 4, 9], [9, 8, 0]])
b = Matrix(content=[[3, 4], [9, 8], [0, 0]])
print(a.transpose() + a)
print(b)
