import numpy as np


def Gsselm(AA, row, X, SINGULAR=False):
    """
    parameters:

        REAL*8 , INTENT(IN)  ::  AA(ROW,ROW+1)

        INTEGER, INTENT(IN) :: row

        REAL*8 , DIMENSION(row),INTENT(out) :: X

    output:
    """
    A = AA
    EPS = 3.0e-15  # EPS=3.0D-15
    swap_ik = np.zeros(8)  # Whole vector initialized to zero
    tmp = 0.0

    # Check dimensions of input matrix

    # gaussian elimination
    # row reduction of matrix

    # fortran indices start at 1 and the stop value seems to be inclusive:
    #
    # do k=1, row-1 total of row-1 operations
    # becomes
    for k in range(0, row):
        # Pivotal strategy - SWAP rows to make pivotal element a[k][k] have the
        # greatest magnitude in its column. This prevents unnecessary division by
        # a small number.
        for i in range(k, row + 1):
            if abs(A[i][k]) - abs(A[k][k]) > EPS:
                # If pivotal element is not
                # the highest then
                # swap i'th and k'th rows
                for j in range(k, row + 2):
                    swap_ik[j] = A[k][j]
                    A[k][j] = A[i][j]
                    A[i][j] = swap_ik[j]
        # If the Matrix is SINGULAR then EXIT program
        if abs(A[k][k]) < EPS:
            SINGULAR = True
            return SINGULAR, X

        # Perform row-reduction with pivotal element a[k][k]
        for i in range(k + 1, row + 1):
            # range/indexing could easily be wrong here
            for j in range(row, k, -1):  # starting from end of column
                A[i][j] = A[i][j] - A[k][j] / A[k][k] * A[i][k]

    # At this point, the bottom triangle is Zero

    # Back Substitution - Solutions of equations
    X[row] = A[row][row + 1] / A[row, row]
    for k in range(row - 1, 1, -1):  # more reverse indexing
        tmp = 0.0
        for j in range(k + 1, row):
            tmp = tmp + A[k][j] * X[j]
        X[k] = (A[k][row + 1] - tmp) / A[k][k]

    return SINGULAR, X
