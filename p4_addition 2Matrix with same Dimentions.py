def print_matrix(mat: dict):
    result = []
    for r in mat.keys():
        c = mat[r]
        for c in mat[r].keys():
            t = mat[r][c]
            print("(", r, ", ", c, ", ", t, ")")
            result.append((r, c, t))

    return result

def input_matrix():
    n = int(input("enter number of rows: "))
    m = int(input("enter number of columns: "))

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

def add(mat1 , mat2):
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

mat1 = input_matrix()
mat2 = input_matrix()
nmat = add(mat1, mat2)
print_matrix(nmat)
