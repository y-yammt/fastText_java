package cc.fasttext;

import java.io.IOException;

import ru.avicomp.io.FTInputStream;
import ru.avicomp.io.FTOutputStream;

/**
 * TODO: implement
 * See <a href='https://github.com/facebookresearch/fastText/blob/master/src/qmatrix.cc'>qmatrix.cc</a> &
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/qmatrix.h'>qmatrix.h</a>
 * <p>
 * Created by @szuev on 24.10.2017.
 */
public class QMatrix extends Matrix {

    public boolean qnorm_;
    public int codesize_;
    //uint8_t* codes_;
    public byte[] codes_;
    //uint8_t* norm_codes_;
    public byte[] norm_codes_;

    //std::unique_ptr<ProductQuantizer> pq_;
    public ProductQuantizer pq_;
    //std::unique_ptr<ProductQuantizer> npq_;
    public ProductQuantizer npq_;

    public QMatrix() {
    }

    /**
     * <pre>{@code QMatrix::QMatrix(const Matrix& mat, int32_t dsub, bool qnorm)
     *  : qnorm_(qnorm), m_(mat.m_), n_(mat.n_),
     *  codesize_(m_ * ((n_ + dsub - 1) / dsub)) {
     *  if (codesize_ > 0) {
     *      codes_ = new uint8_t[codesize_];
     *  }
     *  pq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer(n_, dsub));
     *  if (qnorm_) {
     *      norm_codes_ = new uint8_t[m_];
     *      npq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer(1, 1));
     *  }
     *  quantize(mat);
     * }}</pre>
     *
     * @param mat
     * @param dsub
     * @param qnorm
     */
    public QMatrix(Matrix mat, int dsub, boolean qnorm) {
        //TODO:
    }

    /**
     * <pre>{@code void QMatrix::save(std::ostream& out) {
     *  out.write((char*) &qnorm_, sizeof(qnorm_));
     *  out.write((char*) &m_, sizeof(m_));
     *  out.write((char*) &n_, sizeof(n_));
     *  out.write((char*) &codesize_, sizeof(codesize_));
     *  out.write((char*) codes_, codesize_ * sizeof(uint8_t));
     *  pq_->save(out);
     *  if (qnorm_) {
     *      out.write((char*) norm_codes_, m_ * sizeof(uint8_t));
     *      npq_->save(out);
     *  }
     * }}</pre>
     *
     * @param out {@link FTOutputStream}
     * @throws IOException if an I/O error occurs
     */
    @Override
    void save(FTOutputStream out) throws IOException {
        //TODO:
    }

    /**
     * <pre>{@code
     * void QMatrix::load(std::istream& in) {
     *  in.read((char*) &qnorm_, sizeof(qnorm_));
     *  in.read((char*) &m_, sizeof(m_));
     *  in.read((char*) &n_, sizeof(n_));
     *  in.read((char*) &codesize_, sizeof(codesize_));
     *  codes_ = new uint8_t[codesize_];
     *  in.read((char*) codes_, codesize_ * sizeof(uint8_t));
     *  pq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer());
     *  pq_->load(in);
     *  if (qnorm_) {
     *      norm_codes_ = new uint8_t[m_];
     *      in.read((char*) norm_codes_, m_ * sizeof(uint8_t));
     *      npq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer());
     *      npq_->load(in);
     *  }
     * }
     * }</pre>
     *
     * @param in {@link FTInputStream}
     * @throws IOException if an I/O error occurs
     */
    @Override
    void load(FTInputStream in) throws IOException {
        //TODO:
    }
}
