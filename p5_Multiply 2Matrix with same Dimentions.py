def print_matrix(mat: dict):
    result = []
    for r in mat.keys():
        c = mat[r]
        for c in mat[r].keys():
            t = mat[r][c]
            print("(", r, ", ", c, ", ", t, ")")
            result.append((r, c, t))

    return result


def input_matrix(n, m):
    vector = dict()
    for r in range(n):
        for c in range(m):
            t = int(input("(" + str(r) + ',' + str(c) + '): '))
            if t != 0:
                if r not in vector:
                    rdict = dict()
                else:
                    rdict = vector[r]
                rdict[c] = t
                vector[r] = rdict

    return vector


def transpose(mat):
    result = dict()
    for r in mat.keys():
        c = mat[r]
        for c in mat[r].keys():
            if c not in result:
                rdict = dict()
            else:
                rdict = result[c]
            rdict[r] = mat[r][c]
            result[c] = rdict
    return result


def add(mat1, mat2):
    ans = mat1;

    for r in mat2.keys():
        c = mat2[r]
        for c in mat2[r].keys():
            if r in ans:
                if c in ans[r]:
                    ans[r][c] += mat2[r][c]
                else:
                    ans[r][c] = mat2[r][c]
            else:
                ans[r][c] = mat2[r][c]

    return ans;


def get_element(mat, i, j):
    if i not in mat:
        return 0
    if j not in mat[i]:
        return 0
    return mat[i][j]


def multiple(mat1, mat2, n, m, z):
    result = dict()

    for i in range(n):
        for j in range(z):
            for k in range(m):
                val = get_element(mat1, i, k) * get_element(mat2, k, j)
                t = get_element(result, i, j) + val
                if i in result:
                    rdict = result[i]
                else:
                    rdict = dict()
                rdict[j] = t
                result[i] = rdict

    return result


n = int(input("enter number of rows first mat: "))
m = int(input("enter number of columns first mat: "))
print("number of rows second mat is ", m)
z = int(input("enter number of columns second mat: "))

print("-----First Matrix-----")
mat1 = input_matrix(n, m)
print("-----Second Matrix-----")
mat2 = input_matrix(m, z)
nmat = multiple(mat1, mat2, n, m , z)
print_matrix(nmat)
